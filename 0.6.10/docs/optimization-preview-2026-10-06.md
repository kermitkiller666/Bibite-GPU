# October 6 optimization preview

This is a separate, local 0.6.10 development preview. It does not update a
running game, replace the previous portable previews, or constitute a stable
release. The underlying public game assets remain 0.6.3.1, not 0.6.4.

## Implementation status

| Audit item | Implemented | Remaining boundary |
| --- | --- | --- |
| Food sensing | Cell-local segmented overflow lists replace both whole-pellet fallbacks. A parallel prefix/count/scatter builds contiguous cell ranges. Dense overflow uses a warp per Bibite and cached query results; ordinary worlds retain scalar sensing. Full-native/imported brain input preparation shares a due food query with generic perception. Rotation/normalization happens only for the winning targets. | Sensing cadence remains part of Exact/Extreme behavior; no claim of exact stock biological parity. |
| Neural kernels | Specialized sparse and dense 34 → 12 → 12 → 15 evaluators; staged CUDA Graph pipeline; phase-aware eight-entry graph cache; opt-in FP16 Tensor Core prototype. | Standard fused is the default. Tiny independent, evolving brains do not automatically benefit from a shared-weight matrix multiplication. |
| Live work | Compact living-index lists; births append, deaths compact; grids and launch geometry follow actual population/food rather than population reservation. | Reserved genome arrays still consume capacity-dependent VRAM. This is not a completely compact genome allocator. |
| Invariant math | Cached per-Bibite digestion efficiency and world drag retention, invalidated on relevant changes and reconstructed after load. | Movement prediction retains effect ordering rather than reusing a stale velocity across contacts/grabbing. |
| Contacts | Ordinary shared-memory unique-pair tiles stay enabled alongside local overflow/large-body handling. Crowded overflow chains are loaded in warp-sized shared tiles. Dense home tiles become independent warp tasks, rather than assigning a whole crowded cell to a single warp. | Tiny cells retain the guarded fallback. This is not an unlimited dense all-pairs solution. |
| Presentation | Warp reductions/compaction; viewport-only detail downloads; slower chart summaries; one property block per textured body; independently owned CUDA render snapshots and separate render stream. | Original sprite parts are still CPU GameObjects. Full GPU-atlas/instanced original textures are **not implemented**. Unity GUI/FPS validation of this preview is outstanding. |
| Saves/storage | Compact imported-brain pool, reusable bounded pinned/GPU staging, independently owned checkpoint blob, asynchronous transactional disk write, live-row format 12. | Capturing a coherent save still briefly pauses the simulation worker. No zero-pause saving promise. Pool capacity follows its historical peak. |
| Accounting | Full simulation-call time, timed GPU interval, separate graph food/native-brain/feeding phases, graph-build time/cache hits, and snapshot cost. | In-game end-to-end FPS/warp must still be measured. Kernel share is not GPU utilization or the fraction of all game work on GPU. |

## Kernel and precision choices

New native brains retain FP16 genetic weights/biases and FP32 accumulation;
physics stays FP32. Sparse connection masks control whether a gene contributes.
Inactive genes are not read/multiplied, including dormant NaNs: disabling a
connection has effective weight zero without destroying the genetic value.

The normal fused kernel uses the compiler's standard register budget. A
higher-occupancy launch variant and smaller body-work layouts are available for
diagnostics, but higher occupancy did not consistently improve throughput.
Imported brain instances use the staged graph because their topology differs.

GPU Settings exposes fused, sparse-graph, and dense-graph choices for the **next
simulation**. A loaded checkpoint retains its saved diagnostic configuration.
The Tensor Core path is opt-in through the engine CLI, not a recommended setting;
it is rejected on unsupported hardware and retains the non-Tensor default path.
Compiled Ada disassembly contains `HMMA.16816.F32` for the prototype. This proves
instruction use, not a speedup. Padding a per-Bibite matrix-vector calculation
leaves many tensor operations unused; every Bibite has different evolving weights.

The engine CLI accepts `--kernel fused|sparse|dense|tensor`,
`--fused-occupancy standard|high`, and `--body-layout flat|warp|half`.

## Checkpoints and compatibility

Format 12 writes living body/genome rows and actual imported-instance rows,
while retaining public slot mappings, free queues, RNG state and full slot
incarnation counters. Dead-slot counters matter too: reusing a slot must not
silently recreate the same inspector/grab identity after a save/load.
Validation uses separate scratch storage rather than overwriting these IDs.

Legacy checkpoints are read and their expanded imported-instance payload is
streamed into the compact pool. **Older binaries cannot read format 12.** Back
up old worlds, keep the old EXE, and save preview worlds under new names.

One native-only fixture with two living Bibites, an 8192-slot population reserve
and 32 food slots uses about 260 KB instead of 18.5 MB in the expanded legacy
format. This roughly 71-fold reduction is a sparse-fixture result, not a promise
for a full population or every save.

## Verification and measured limits

Managed/Core builds and the focused optimized native suite have passed. Tests
cover local crowded-food counts/nearest targets, sparse/dense/Tensor brain
parity, mutation/dormant links, graph-cache phases/settings invalidation,
compact imported-instance growth/reuse, live-only checkpoints and owned render
captures. A D3D11 headless test overlapped stepping with delayed render callbacks
without reading the mutable simulation state.

The final full native regression suite passed on the RTX 4070 Ti; its log is
`work/verification/optimized-full-tests-20261006s.log`. It includes dense-food
warp versus scalar-reference parity and independent dense contact task dispatch,
corrupted
checkpoints, live/dead incarnation preservation, eight crowded save/reload/resume
cycles and five simulated minutes of large-map ecology. Do not infer a successful Unity save-menu test from a
successful native checkpoint test. The in-game ZIP wrapper, GUI responsiveness,
30-FPS original-texture behavior, and final shutdown need a separate authorized
Unity test. An existing isolated test game was left untouched.

Final three-repeat, 256-tick Extreme-mode comparisons on CUDA device 0
(RTX 4070 Ti) showed the standard fused build's median full-call warp approximately:

| Scenario | Previous native DLL | Standard fused candidate |
| --- | ---: | ---: |
| 256 founders, 65536-slot reserve, 8192 food slots | 209× | 268× |
| 2048 founders, large world, 8192 food slots | 273× | 281× |
| 2048 founders, denser world, 32768 uniformly distributed food slots | 179× | 154× |

The uniform high-food case was roughly 14% slower in this final sweep: this
build does **not** improve every workload. All short uniform comparisons retained
the same living counts and pellets-eaten totals; their timing ranges overlap in
some scenarios. Do not substitute earlier intermediate-build numbers for these.

Separate 2048-founder fertile-zone tests with 32768 food slots had median full
call times of about 787 → 331 ms (zone radius 20) and 710 → 119 ms (radius 100),
approximately 2.4× and 6.0× improvements. Individual records include phase times.
The same populations survived the short tests, but pellet totals vary with
parallel consumption order; this is not a long-run deterministic parity claim.
The rejected pointer-chain/serial-per-Bibite overflow attempts and their severe
stress-case regressions were kept as evidence, not shipped as the final engine.

These are short **headless engine** runs, not graphical game speeds. Another
project was using the same 4070 Ti, and repeat timing varied substantially.
Population/food outcomes matched in these short comparisons. The 5060 Ti was
not used. Neither these data nor phase-bypass diagnostics support a 1000× or
2500× unchanged-gameplay promise. The Tensor, sparse/dense graph, occupancy and
body-layout experiments varied by workload and did not justify replacing the
fused default universally.

Longer ecology remains a separate correctness problem: the final five-minute
large-map regression retained 151 of 512 founders with 7 births and 368
starvation deaths. Faster
evaluation does not establish healthy evolution or exact stock energetics.

## Local portable artifact

`work/portable-0.6.10/Bibites GPU Fork 0.6.10 - OPTIMIZED PREVIEW.exe`
is the separate 72,923,136-byte portable launcher. Its SHA-256 is
`EE6C06FCAE44F4CFA788E619A150FB69675B25113A3A4CDFA13E952CDAFDEBB8`.
It contains 243 entries, including the matching three canonical plugin DLLs;
the embedded payload hash and individual DLL hashes were checked. The launcher
also checks these DLL hashes when reusing its extracted folder, preventing a
stale companion library from silently being mixed with this engine.

The native DLL SHA-256 is
`1BFE26B151EA12D470DA97C10A2133F2B3C498D6E1A98F5321BE6B26C37278F7`.
No personal configuration/saves/logs/screenshots are bundled. Previous PARITY
and SAVE FIX preview EXE hashes remain unchanged. This artifact was not launched
for Unity verification and was not pushed/uploaded to GitHub in this operation.

## Reproducible checks

Use matching managed/native libraries. Build the CUDA world tests and run the
full suite; focused entry points include `--optimizations-only`,
`--kernel-parity-only`, `--checkpoint-only`, `--biology-only`, and
`--brain-io-only`. Run performance comparisons sequentially, never while a
different test from this project is using the GPU, and report other-device load.
Keep the baseline executable/library hashes and individual run JSON files.
Never treat a diagnostic that skips a simulation phase as a valid speedup.
