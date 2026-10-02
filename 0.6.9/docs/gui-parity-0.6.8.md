# GUI parity checklist — GPU fork 0.6.8

Source audit: 2026-10-01. Reference UI: the supplied **The Bibites 0.6.3.1** build. This is a BepInEx/CUDA adaptation, not full original-simulation parity. Stock CPU worlds retain their original menus and biology.

## Restored or adapted in native GPU worlds

| Original interface | Current implementation and boundary |
| --- | --- |
| Selected-Bibite toolbar and Stats (`1`) | Original panel roots, fonts/art and tab toolbar; opaque content, scrolling, drag/close and sticky Follow/Species/Deselect actions. Native generation, energy, age, size, motion and position. |
| Genes (`2`) | Native size, speed, turn speed, metabolism, lifespan, mutation strengths and colour. Not the complete original genome. |
| Biology (`3`) | Native energy/lifecycle, motion outputs, food/neighbour sensing and pheromone outputs. No fabricated organ measurements. |
| Brain / Expanded Brain (`4` / `5`) | Native topology graph and paged node/synapse readback. Imported-brain state is displayed; compact sensor/pre-activation values that are not retained are labelled accordingly. |
| Individual actions | Tag edit/apply/copy, remove, follow and forced reproduction. Mutation commands validate the selected slot's incarnation before acting; queued/completed/failed feedback stays above the footer. “Lay egg” creates offspring directly; there is no incubating egg body. |
| Information panel | Full-world Bibite/pellet counts and energy; clickable lineage/tag rows. At large populations, row counts/energy are sample estimates (`~`), not an exact census of every lineage. |
| Species panel | Living-lineage list with member selection. Preserves imported lineage identity; does not claim automatic species formation or a species tree. |
| Rectangle selection | Captured Bibite membership/energy and member inspection. Uses available presentation samples, excludes food, and displays at most 256 member buttons. |
| Historical charts | Original navigation/graph renderer for population, energy, birth/death counts, sampled age and brain size. Age uses the stock hours axis. History is snapshot-sampled, not every simulation tick. |
| Settings | Original Graphics/Game/Hotkeys pages retained; GPU Settings entry added. GPU device, population, texture budget and Extreme profile apply to subsequent worlds; graphics FPS applies live. |
| Dynamic Settings | Food density changes target pellet count; fertility changes regrowth; pellet energy changes food value. Other stock settings are not all connected to the compact model; a notice explains this. |
| Save / Load / QuickSave / QuickLoad | Stock save interface plus paired native checkpoint; preflight/error handling and staged pair commit. Friendly lineage/tag names are stored in `scene.gpuPresentation`. Keep the wrapper and checkpoint together. |
| Input / modal behaviour | Native selection respects UI hit-testing; typing and settings block simulation hotkeys. Inspector/selection visuals suspend under fullscreen charts, modal panels and hidden UI. R/G/O select from available snapshot members, not an exhaustive large-world search. |

## Explicitly unavailable in GPU mode

- Save individual Bibite / edit selected live Bibite: visible disabled controls explain that a complete stock genome cannot be reconstructed. Save World preserves the native individual. The original template editor remains for templates placed later.
- Parents, children, per-individual eggs-laid history, original stomach/organs/combat damage and species-distance fields: not recorded or not simulated.
- Separate egg selection (`E`): explains the direct-offspring lifecycle. Bulk rectangle lay/remove/tag is disabled because sampled recycled slots are not a persistent safe group.
- Gene-distribution charts, age-at-death distribution and per-Bibite eggs-laid history: unavailable notices replace misleading zero plots.
- Spatial zone effects, stock radiation/pheromone towers and many stock physics/biology/mutation controls: not native-world features. A restored menu is not evidence that its original simulation subsystem exists.
- Structural brain-topology evolution and automatic stock speciation remain engine limitations; the GUI does not invent either.

## Save-name metadata lifecycle

Scene reset and stock-world loading clear old lineage/tag name maps. Native loading validates the sidecar, stops the old world, then imports the new wrapper's names. Native startup does not clear those imported names. Missing metadata clears maps and falls back to ID-based labels. Names are plain text, not rich-text instructions. Import/export limits each map to 65,536 entries and each name to 100 UTF-16 code units without splitting a surrogate pair.

## Validation boundary / remaining checks

Integrator-observed: Stats/Genes/Biology/Brain/Expanded Brain worked with direct graphics interop disabled; panel styling passed visual inspection; paused tag application worked and metadata was written into the save. The repaired render-thread path then passed live Stats/Brain/chart interaction, native save/load with tag recovery, and paused species browsing at the 131,072 population cap. These checks do not validate every original menu.

Automated metadata/action tests pass: exact unsigned 64-bit IDs, culture-independent round-trip, absent/malformed metadata, size limits, surrogate-safe names, independent exported data, and action-message routing without slot-prefix/offspring/tag collisions. The managed build passes with no warnings or errors. These tests do not exercise Unity layout or native graphics interoperability.

- [x] Repeat Stats/Brain/chart and selection checks with shared rendering, including pause and high warp. The final 30-second high-warp sample held 59.8 FPS with no measured frame over 50 ms.
- [x] Reload a tagged native individual through the normal UI and confirm its name. Metadata clearing across absent/stock metadata is also covered by pure regression tests.
- [x] Native Save Game/Load Game and scene replacement preserve population and metadata. Final application exit releases GPU resources before Unity graphics teardown and the process exits.
- [ ] Retry rectangle dragging for at least 0.5 seconds (stock input requires more than 0.15 seconds); its visible group panel has not yet been confirmed in the runtime smoke test.
- [ ] Exercise native-to-stock same-scene QuickLoad, every imported-template/editor path, destructive save-menu actions on disposable fixtures, and all window sizes.
- [ ] Prolonged/overnight renderer stability. The earlier multithread-protected candidate failed; the replacement render-thread handoff passed the bounded live runs documented in `gui-stability-2026-10-01.md`, not an overnight soak.

Implementation: [inspector](../managed/GpuNativeInspector.cs), [charts](../managed/GpuNativeCharts.cs), [world/UI bridge](../managed/GpuNativeWorldBridge.cs), [menu integration](../managed/GpuMenuIntegration.cs), [save/settings hooks](../managed/Plugin.cs), [name metadata](../managed/GpuPresentationMetadata.cs).
