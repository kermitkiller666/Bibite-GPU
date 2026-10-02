# Bibites GPU Fork 0.6.9 (release candidate)

Experimental CUDA acceleration runtime for The Bibites 0.6.x.

This repository is the mod source, not a copy of The Bibites. It does not
include the base game, Unity assets, BepInEx binaries, saves, or third-party
assemblies. Obtain a compatible copy of The Bibites separately. See
[COMMUNITY_RELEASE.md](COMMUNITY_RELEASE.md) before publishing or installing.

## 0.6.9 feeding change

GPU-native Bibites now take portions from pellets over multiple simulation
updates. Pellets visibly shrink until depleted, and a per-Bibite stomach caps
intake while digestion releases energy gradually. This is a compact GPU
approximation, not a port of every stock mouth, stomach, diet and organ rule.
Version-3 GPU checkpoints preserve partial pellets and stomach contents; the
loader upgrades earlier version-1/2 checkpoints. Back up saves before trying
this release candidate, because older plugin binaries cannot read v3 saves.

This release runs a compact evolving world as structure-of-arrays data on one
NVIDIA GPU while retaining the original Unity menus, camera, HUD, settings,
selectable 10-60 FPS graphics, procedural Bibite sprites, pellet texture, placer, and
Information and Statistics panel.

The supplied and packaged base game is **The Bibites 0.6.3.1 Windows x64**.
The code has not been rebased against 0.6.4 because no 0.6.4 game assembly was
provided. It must not be described as an exact 0.6.4 fork yet.

## 2026-10-01 menu and stability repair (still 0.6.8)

The historical local 0.6.8 ready-to-run test bundle launched through
`START BIBITES GPU.exe`. This source-only 0.6.9 folder does **not** contain
that launcher or the game. After building and installing the mod into your
own compatible game copy, use Settings > GPU Settings to check the selected
device and world options.

- Restored the original Stats, Genes, Biology, Brain and Expanded Brain panel
  shells, toolbar controls and original-style species/tag browsing. Available
  values come from the native simulation; missing stock-model data is labeled,
  not invented. Large-world rectangle selection is a sampled, read-only list.
- Added GPU Settings to the original settings navigation. Its opaque dialog
  fits the screen, scrolls, keeps Close visible, and blocks clicks/shortcuts
  from leaking into the world. Escape closes the GPU dialog before its parent.
- Selection, Follow, tagging and queued actions work through CPU menus while
  the GPU remains authoritative. Action results stay visible, including
  insufficient-energy failures; command identity checks avoid stale-slot edits.
- Save previews use GPU counts. Friendly tags/lineage names survive save/load.
  Wrapper and checkpoint files are staged before replacement, with rollback
  for ordinary I/O failures. Missing GPU companions are rejected explicitly.
- Stock-save loading now selects compatibility mode for that session without
  changing the preference for future worlds. The native-mode checkbox cannot
  replace a running CPU world, and leaving/restarting a world resets its state.
- Hardened checkpoint bounds/validation, paused command processing, resource
  ownership and CUDA/Direct3D synchronization. Shared-buffer work is scheduled
  on Unity's render thread; a fallback renderer remains available.
- Texture-state refresh follows the living presentation sample, not unused
  population capacity. Quit waits nonblockingly for pending saves and render
  cleanup before Unity tears down graphics; uncertain interop failures disable
  further shared-buffer attempts for the process.

Current controls:

| Control | Action |
| --- | --- |
| Ctrl + Page Up / Page Down | Increase / decrease requested time warp |
| Page Up / Page Down | Original camera movement-speed controls |
| F5 / F9 | Quick-save / quick-load |
| Space | Pause / resume |
| 1-5 | Selected Bibite inspector tabs |
| Ctrl + Shift + F8 / F9 / F10 | Stock-compatibility validation / brain-mode shortcuts |

Population, CUDA-device choice, native mode, texture limit and Exact/Extreme
profile apply to the next world. Graphics FPS applies live. In Dynamic Settings,
food density, fertility/regrowth and pellet energy affect the running native
world; the original zone, tower, organ, combat and mutation options are not all
implemented by the compact engine. The menus explain these boundaries.

The repair passed all managed regressions and both native CTest tests.
On the RTX 4070 Ti, the final 1x smoke sample held 60 FPS (17.6 ms maximum
frame). The requested-1000x sample held 59.8 FPS (31.7 ms maximum), ending
at 79,796 living Bibites and 142.2x achieved speed. Neither 30-second sample
had a measured frame over 50 ms. The population later reached 131,072;
native reload, tag recovery and clean process exit passed.
These are bounded smoke tests, not an overnight stability guarantee or
matched throughput benchmarks. Detailed scope and remaining checks:
[`docs/gui-stability-2026-10-01.md`](docs/gui-stability-2026-10-01.md).

## Historical release notes

The following version notes and performance figures predate this repair.
They are retained as history, not as newly repeated benchmarks. Version labels
below refer to this fork, not to the supplied upstream game's version.

## What changed in 0.6.8

- Stock Dynamic Settings now update GPU food during a running simulation:
  biomass density changes the pellet target, fertility changes regrowth rate,
  and pellet energy changes food energy. New worlds reserve at most 32,768
  pellet slots so density can rise without rebuilding the world. The GPU
  Settings tab shows active food, target, reserve, growth, and energy.
  Older checkpoints retain their original pellet capacity.
- Compact-brain worlds no longer reserve the maximum 256-node / 512-synapse
  placed-brain payload for every possible Bibite. That mutable storage is
  allocated only after a stock/template brain is actually placed.
- Placed brains now share one packed immutable topology record per distinct
  graph. Per-Bibite storage keeps only evolving biases, synapse weights and
  recurrent state; node types, sensor/action identities, enabled-node lists
  and connection endpoints are no longer duplicated for every descendant.
- The native CPU snapshot staging area is capped independently at the same
  8,192-member presentation sample used by the GUI instead of scaling with a
  500,000 simulation cap.
- Direct rendering uses the proven 36-byte FP32 position/colour/UV layout.
  The experimental 20-byte vertex layout was reverted because this Unity/D3D11
  path dropped most distant GPU silhouettes even though close-up sprite proxies
  remained visible.
- The native TimeKeeper now owns the achieved-speed fields continuously. The
  stock 1x presentation clock can no longer overwrite them once per second.
- GPU settings now includes a configurable close-up textured-Bibite limit.
  Presets cover Off, 96, 256, 512, 1,024 and 2,048, with a custom field up to
  2,048. The rest of the population retains the lightweight GPU silhouette.
  The safe packaged default remains 96 because every textured Bibite uses the
  original multi-part Unity procedural sprite renderer.
- GPU-native worlds now save and load through the normal save UI. Each normal
  `.zip` wrapper has a complete `.zip.bgfgpu` companion containing the CUDA
  world, including evolving brains, food queues, pheromones and counters.
  Checkpoint writing runs on the GPU worker so F5 and the save panel no longer
  block Unity's render thread while CUDA/Direct3D owns shared buffers.
- On the RTX 4070 Ti, an otherwise identical 500,000-Bibite headless world used
  about **415 MiB** of additional GPU memory instead of **6,337 MiB** in 0.6.7,
  a measured **15.3x reduction** before a stock/template brain is placed.
- Alternating 4,096-step A/B runs at 500,000 Bibites measured a 0.6.8 median of
  **14.155x** versus **13.975x** for 0.6.7, about **1.3% faster**. At 32,768
  Bibites the five-run medians were **656.354x** and **652.349x**, about
  **0.6% faster**. The primary benefit is capacity and allocation pressure,
  not a large raw-kernel speedup.

## What changed in 0.6.7

- Fixed a CUDA/Direct3D registration timeout race. A timed-out registration can
  no longer continue on the worker after Unity has destroyed its mesh buffers,
  and every queued graphics request is completed safely if the worker exits.
- GPU selection now resets the compatibility brain, vision, pheromone and
  validation contexts together. An already-running native world stays on its
  original GPU safely; the selection applies to the next world.
- Native world creation now rejects NaN/infinite numeric settings and pellet
  counts above the supported 32,768 limit before allocating GPU memory.
- Initialized every capacity-only Bibite record copied by the legacy snapshot
  and statistics paths. NVIDIA memcheck and initcheck now both report zero
  errors instead of exposing uninitialized device bytes from unused slots.
- Added deterministic tests for the five-simulated-second food delay, invalid
  numeric configuration and the pellet hard cap.
- Re-ran the real graphical build at 1x and requested 1000x. Both held 30 FPS
  with no frame over 50 ms; click inspection, species browsing, live food
  counts, the stock Information panel and graceful shutdown all passed.

## What changed in 0.6.6

- Fixed the remaining large-cap graphics freeze. CPU presentation buffers now
  hold at most 8,192 Bibites even when the selected simulation cap is 500,000,
  so Mono no longer marshals a half-million-entry array during each snapshot.
  The headline population and GPU draw count remain exact.
- Direct CUDA/Direct3D refreshes now write only render vertices and counters;
  they no longer repack unused CPU Bibite/pellet records or total-energy data.
- Population-aware simulation batches bound how long the render handoff can
  wait as a world grows. The handoff budget follows the selected graphics FPS,
  preventing consecutive refreshes from being cancelled mid-batch.
- Eaten plant pellets now remain absent for five simulated seconds before a
  compact GPU timing wheel regrows them. The food count therefore falls when
  food is eaten and rises when food regrows instead of remaining artificially
  constant.
- A selected Bibite is retained across sampled snapshots at populations above
  8,192 instead of closing merely because that slot was absent from one sample.
- In a 500,000-cap, 8,192-starter test at requested 1x, the engine held 1.0x,
  the worst UI frame was 33.7 ms, no frame exceeded 50 ms, and the largest CPU
  snapshot was 1.10 ms. At requested 1000x and 79,101 living Bibites, the worst
  completed graphics-update gap was 50.12 ms and the worst UI frame was 34.1 ms.

## What changed in 0.6.5

- Removed the one-frame 1x speed flash. The stock speed readout now keeps the
  selected native multiplier while Unity's presentation clock remains safely
  decoupled at 1x.
- Direct CUDA render refreshes use a short, cancellable render-safe handoff
  instead of waiting behind long simulation or snapshot work, and simulation
  batch length adapts to the selected graphics cadence.
- Large-capacity rendering now submits only living Bibites and active pellets.
  CUDA clears the full shared mesh once, then clears only the tail that became
  unused instead of rewriting a 500,000-Bibite capacity buffer every frame.
- Reduced the distant Bibite LOD from detached body, fin and eye triangles to
  one connected nine-vertex silhouette. Close-up stock procedural sprites are
  unchanged.
- Clicking a GPU Bibite now opens a CPU-side inspector with the familiar Stats,
  Genes, Biology, Brain and Expanded Brain choices, plus a visible selection
  reticle. Its values come from the authoritative CUDA Bibite at 10 Hz; complete
  node and synapse data crosses to the CPU only while a Brain tab is open.
- The inspector can follow the selected Bibite, change its tag, request an
  offspring or remove it. Those actions are queued back to the GPU worker so UI
  clicks never mutate a stale CPU copy.
- The stock Species button now opens a GPU-backed living-species browser, and
  the Information panel creates and updates its top-species rows from native
  lineage data.
- Large-population chart and species presentation is calculated from one
  cached, evenly distributed 8,192-Bibite sample. Counts remain exact through
  8,192; above that, overall population is exact while lineage distributions
  and age/brain summaries are estimates. This reduced chart work; version 0.6.6
  also caps the underlying P/Invoke arrays, removing the remaining periodic
  large-cap snapshot freeze.
- Removed the enormous temporary CPU vertex allocation during large-cap mesh
  creation; CPU fallback arrays are allocated only if direct interop fails.

## What changed in 0.6.4

- Fixed the large-map ecology collapse. Food vision now scales with map area
  and pellet density, while large-world mobility and capture range are bounded
  together so accelerated Bibites cannot step over or orbit food.
- Added a low-energy foraging reflex. It fades out after feeding, leaving the
  neural brain and evolution in control during normal operation, but gives
  randomly generated starter brains the same basic viability as curated stock
  starter species.
- Replaced the benchmark-only 4-12 minute procedural lifespan with a 30-90
  minute starter range. Projected stock-template lifespans now use the same
  useful scale while remaining heritable and mutable.
- Starvation, age and invalid-state deaths are counted separately and displayed
  live in the GPU overlay. Their sum is checked against total deaths in the
  native invariant suite.
- Added a permanent 15,000-half-extent, five-simulated-minute regression test.
  It checks food consumption, population survival, lifespan and finite state.

## What changed in 0.6.3

- Native simulations bypass the stock game's synchronous pellet seeding and
  stock Bibite spawner before the GPU world takes ownership. Large worlds no
  longer create and then destroy hundreds of thousands of Unity objects.
- Compatibility brain, vision and pheromone GPU batches are suppressed during
  native-world startup, removing the unnecessary 262,144-entity vision buffer.
- Native startup begins on the frame after the simulation scene is ready rather
  than waiting 0.75 seconds, and the log reports handoff and cleanup timings.
- A measured 15,000-extent regression run reached authoritative GPU ownership
  in 150 ms, cleaned one temporary Bibite and pellet in 3.3 ms, and used about
  2.03 GiB of private process memory instead of the previous roughly 5.4 GiB.

## What changed in 0.6.2

- Selectable time warp now extends through 2,500x, 5,000x and **10,000x**.
- The stock slider, integration controls and native viewer use the extended
  ladder. The graphical keyboard binding is now Ctrl + Page Up / Page Down.
- The high settings remove the old 1,000x throttle. If the world cannot attain
  the selected rate, the engine runs flat-out and the HUD reports actual speed.

## What changed in 0.6.1

- The native population hard ceiling is now **500,000 Bibites**. Values above
  it are rejected before allocation.
- `GPU settings` now has cap presets from 512 through 500,000 plus editable
  custom cap and starting-population fields.
- The chosen population cap is the soft cap for that world and is also the
  allocation size. A 2,048-cap world still allocates for 2,048, not 500,000.
- The settings screen estimates the selected world's GPU allocation before the
  next simulation is started.
- Direct GPU rendering keeps the selected graphics rate at large capacities,
  while CPU-only chart, selection and detail snapshots automatically fall to
  4 Hz above 32,768 and 1 Hz at 100,000 or more.

## What changed in 0.6.0

- Every native batch now reports measured prepare, spatial-index, contact,
  decision, motion and lifecycle GPU times. The same breakdown is visible in
  the in-game overlay.
- Contact and sensing hashes clear only cells occupied in the previous pass;
  the engine no longer clears all 16,384 contact cells every tick.
- Pellet consumption and reproduction use compact GPU queues. Pellet recycle
  and Bibite free-slot allocation no longer scan their full capacity arrays.
- Pheromone diffusion runs every four fixed steps with mathematically adjusted
  diffusion and decay, preserving the elapsed-time response with one quarter
  of the full-grid updates.
- Contact pairs are evaluated once using warp-local shared-memory cell tiles;
  symmetric forces are accumulated for both Bibites. Overflow, oversized-body
  and unusually small-cell worlds retain the exact legacy fallback.
- Every full perception refresh schedules Bibites in spatial-cell order so
  nearby lanes reuse candidate data. Perception and the following decision pass
  share the same ordered phase, avoiding an extra whole-grid barrier.
- CUDA writes persistent Unity Direct3D 11 vertex buffers directly on supported
  NVIDIA systems. CUDA/Direct3D map-unmap synchronization and a short,
  cancellable handoff protect the shared resource, and the standard FP32 CPU
  renderer remains the automatic fallback.
- Direct rendering runs at the chosen graphics FPS while CPU chart/selection
  snapshots run at 10 Hz for normal worlds, 4 Hz above 32,768 capacity, and
  1 Hz at 100,000 or more. This removes the old per-frame mesh upload and keeps
  large reserved capacities from turning GUI snapshots into the bottleneck.
- Dense neural weights are stored as coalesced FP16 pairs and stock-template
  node/synapse arrays are topology-major across Bibites. Brain math keeps FP32
  accumulation.
- Contact grids are double buffered and collision work uses four independent
  eight-lane shared-memory tiles per warp.
- GPU settings now offers `Exact` and `Extreme` profiles. Extreme retains food
  targets, updates the collision grid every four ticks, solves contact every
  eight ticks, refreshes full vision every 160 ticks and runs brains every
  eight ticks. Forces and recurrent-neuron elapsed time are cadence-adjusted.

## Historical performance measurements

All figures below are historical measurements on GPU 0, an NVIDIA GeForce
RTX 4070 Ti. They were not all rerun for the 2026-10-01 GUI/stability repair.
Population and pellet counts are stated for each non-default run.

- Version 0.6.8 memory-layout A/B, 500,000 Bibites / 8,192 pellets / Extreme:
  **14.155x median** versus **13.975x** in an identically rebuilt 0.6.7
  baseline, about **1.3% faster** over three alternating 4,096-step pairs.
- The same 500,000-cap headless allocation consumed about **415 MiB** of added
  GPU memory versus **6,337 MiB** for 0.6.7, a **15.3x reduction**. A graphical
  world additionally owns Unity/CUDA render buffers and engine overhead.
- Version 0.6.8 32,768-Bibite / 8,192-pellet Extreme A/B: **656.354x median**
  versus **652.349x**, about **0.6% faster** across five paired runs.
- The corrected 0.6.8 renderer returns to the previously verified FP32 vertex
  layout so the full GPU population and stock pellet texture remain visible.

- Version 0.6.7 graphical 1x regression, 32,768 cap / 512 starters: **1.0x
  achieved**, **30.1 FPS**, 33.3/33.6 ms p95/worst frame, zero frames above
  50 ms and a 1.45 ms maximum CPU snapshot.
- Version 0.6.7 requested-1000x UI regression reached the full **32,768** cap
  at **30.0 FPS**, with a 33.6 ms worst frame, zero frames above 50 ms, a
  1.55 ms maximum CPU snapshot and a 7.95 ms maximum simulation batch. The
  captured state showed 3,823 active plants and 1,613,456 feeding events, so
  the displayed food total was changing under load.

- Version 0.6.6 graphical freeze regression, 500,000 cap / 8,192 starters at
  requested 1x: **1.0x achieved**, **30.0 FPS**, 33.3/33.7 ms p95/worst frame,
  zero frames above 50 ms, 1.10 ms maximum CPU snapshot and 75.05 ms maximum
  completed render-update gap.
- Version 0.6.6 high-warp stress run, same cap/start at requested 1000x:
  **79,101 living Bibites**, **211.4x achieved**, **30.0 FPS**, 34.1 ms worst
  UI frame, zero frames above 50 ms, 1.29 ms maximum CPU snapshot and 50.12 ms
  maximum completed render-update gap.
- Food regression captures recorded 8,151 active plants, then 8,089 after
  additional feeding, then 8,110 after delayed regrowth; the total is no longer
  pinned to the configured 8,192.

- Exact profile, 8,192 steps: **1,037.329x real time** and **99.873% GPU share**.
- Extreme profile, two 8,192-step runs: **2,593.840x** and **2,645.086x**, with
  active feeding, births, deaths, sensing, brains and collision response.
- Extreme profile, 1,048,576 steps: **2,231.631x sustained** and **99.936% GPU
  share**. This advanced 7.28 simulated hours in 11.75 wall-clock seconds while
  processing 87,408 births, 87,408 deaths and 2,587,453 feeding events.
- Final graphical smoke test: **990.5x achieved at requested 1000x**, **30.0
  FPS**, live Direct3D buffers, original graphics and an updating Information
  panel; the population then grew normally to its 2,048 cap.
- New 500,000-population Extreme benchmark: **13.632x real time** and
  **99.997% GPU share** with all 500,000 slots alive. Contact handling was the
  main cost at 678.553 ms of the 938.947 ms measured batch.
- New graphical 500,000-cap smoke test: the world started at 512, grew to
  60,145 live Bibites by the capture, rendered at 31 FPS, and kept the original
  Information panel active. The HUD measured about 375x at that point.
- New 256-population 10,000x-target test: **2,298.500x sustained** over
  1,048,576 steps in the native benchmark. The graphical run averaged
  **2,294.3x** over ten seconds and reached about **2,425x at 30 FPS** in the
  final screenshot, with the original Information panel active.
- Before the ecology fix, a 15,000-half-extent Extreme run fell from 2,048 to
  544 Bibites in five simulated minutes, with zero births and only 113 pellets
  eaten. The fixed build retained 1,098 at five minutes, recovered to 3,791 by
  30 minutes, and reached 65,472 of the selected 65,536 cap by one simulated hour.
  That one-hour run recorded 67,647 births, 1,030,010 feeding events, zero
  invalid-state deaths and **810.548x** real time at 65,472 living Bibites.
- The fixed 15,000-half-extent Exact run retained 1,649 of 2,048 starters at
  five minutes, ate 7,142 pellets and ran at **845.222x** real time.
- The player's saved 8,192-starter / 65,536-cap / 8,192-pellet / Exact setup
  was also tested at 15,000 half-extent: 8,007 were alive at five minutes, and
  the population reached its 65,536 selected cap by 30 simulated minutes with
  zero invalid-state deaths.

These are measurements, not a guaranteed speed. Spatial distribution, brain
topology, other GPU work, population, and world size change throughput. Each
warp setting is a target; the HUD shows the achieved multiplier. Short-run
benchmark rates do not establish how long an entire simulated year will take:
evolving brain complexity and spatial clustering can change throughput.

## Graphical mode

Double-click `START BIBITES GPU.exe` in the ready-to-run package. Start a new
simulation with native mode enabled to create a GPU-native world. This folder
preserves the existing player's configuration; the current values are shown
under Settings > GPU Settings. Native graphics FPS is independent of the
requested simulation warp and is a frame-rate target, not a hardware guarantee.

The normal Settings screen contains a `GPU settings` tab for selecting the
CUDA device, population cap, starting population, graphics rate, native mode,
textured-Bibite limit, and Exact/Extreme engine profile. Choose the RTX 4070 Ti
on this PC. Texture-limit changes apply when a new simulation starts;
larger values trade Unity rendering time for more original close-up sprites.
The cap can be typed or chosen from presets up to the 500,000 hard maximum.
It is a per-world soft cap: the engine allocates for the selected value only.
Only the chosen starting population is alive initially.
In the historical 0.6.8 memory-layout test, the 500,000-cap headless allocation
measured about 415 MiB; a graphical world
also reserves roughly 201 MiB of FP32 vertex/index capacity plus Unity and
CUDA overhead. Placing the first stock/template brain lazily reserves up to
about 2.15 GiB more mutable descendant-brain storage at that maximum cap.
There is a dedicated 256-cap preset for the 5,000x and 10,000x targets. Those
values remove throttling rather than guaranteeing the selected rate; the HUD
shows the speed the current ecology actually sustains.

Time-warp choices are exactly:

`1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000x`

`Ctrl + Page Up` and `Ctrl + Page Down` move through those choices. Bare
Page Up/Page Down retain the original camera controls. The normal on-screen
slider remains usable. F5 quick-saves and F9 quick-loads.

The stock Bibite placer is routed into GPU-native storage. Its body data,
colour, generation, mutation strengths, species/tag identity, enabled NEAT
nodes, node functions, biases and FP16 synapse weights are copied to CUDA.
Descendants inherit that topology and mutate supported weights/body traits;
structural topology evolution is not equivalent to the original NEAT model.
The original Information panel
shows live GPU counts, species/tag rows, biomass, age, births/deaths and brain
node/synapse history.

## Precision

- FP32: positions, velocities, forces, biology, neuron state and accumulation.
- FP16: inherited neural and synapse weights.
- 64-bit integers: long-lived counters and identifiers.
- The fixed simulation timestep stays at 0.025 seconds at every time warp.

This mixed policy retains FP32 simulation state while cutting neural weight
traffic; it is not a guarantee of long-run stability. BF16 is not used because
FP16 has sufficient weight range for
the clamped compact model and is the better-supported storage path here.

## Safety and compatibility

Existing stock saves open in compatibility mode. Native GPU worlds now save
through the normal save menu and F5 quick-save. The visible `.zip` stores stock
settings, screenshot, charts and preview metadata; the complete GPU state is
stored beside it as `<name>.zip.bgfgpu` and detected on load. Keep
those two files together when copying a GPU-native save. Deleting a save through
the game also removes its companion. The checkpoint resumes the accelerated
world; it is not a conversion into stock Unity GameObjects, so the original
per-neuron live editor/export workflow still requires compatibility mode.
Checkpoint work is queued to the native worker: simulation briefly pauses at a
safe GPU boundary. Both files are staged before they replace an existing save.
Ordinary I/O failures roll back the prior pair; the two files are not one
power-loss-atomic transaction. Keep backups of important worlds.

Missing companions and invalid headers are rejected before loading. Full
checkpoint payload validation occurs during load, after the old world may have
been retired, so late corrupt-state failures do not preserve an unsaved world.
Save first before switching worlds.

The native ecology approximates stock mechanics that it does not yet model:
meat, grabbing, attacks, full organ biology, directional pheromone behavior,
spatial zone effects and the original tower tools. Native lineage groups are
not the original automatic speciation/family-tree model. Bulk edits and live
per-neuron editing/export are not available for native entities. Above 8,192
living Bibites, selection and many chart/species summaries use a representative
sample while the headline population count remains exact.
Evolution is not bit-identical to the stock game. AMD and Intel GPUs require a
future non-CUDA backend.

## Native engine architecture

The complete native step remains inside one cooperative CUDA launch:

1. sparse-clear and update the cadence-controlled fields and queues;
2. build the contact/sensing spatial hashes;
3. resolve unique tiled contact pairs;
4. refresh spatially ordered perception and evaluate FP16 brains with FP32
   accumulation;
5. resolve movement, feeding and metabolism;
6. reproduce, mutate, die, and recycle queued pellets; and
7. repeat for the requested batch without per-tick CPU transfers.

`Bibites GPU Engine.exe` is the reproducible console benchmark.
`Bibites GPU World.exe` is the engineering viewer. Neither replaces the
graphical game application in the ready-to-run package.

## Build and test

Managed plugin:

```powershell
dotnet build managed\BibitesGpuFork.csproj -c Release
```

Native CUDA backend (from an x64 Visual Studio developer shell):

```powershell
cmake -S . -B build-native-release -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-native-release
build-native-release\BibitesGpuNativeTests.exe
build-native-release\BibitesGpuWorldTests.exe
```

For the 2026-10-01 repair, the managed suite covers time warp, FP16 conversion,
save-file rollback, stock/native session transitions, Windows save-name rules,
bounded presentation-metadata round trips and selected-Bibite action feedback.
Both registered CTest tests passed. Older benchmark and graphical results
above are historical; see the dated repair checklist for final candidate
measurements and any checks still pending.
