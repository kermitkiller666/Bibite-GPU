# Open GitHub issues reviewed 2026-10-04

This is a snapshot of the public tracker, not a claim that the packaged
portable EXE contains the source changes below.

| Issue | Report | Status in this checkout |
| --- | --- | --- |
| [#1](https://github.com/kermitkiller666/Bibite-GPU/issues/1) | AMD support, meat ecology, fertile-zone plant spawning | AMD deferred at the user's request. Meat ecology remains open. Plant spawning now has zone projection and a focused native test. An isolated Unity test displayed Bibites and pellets together on a 3 Islands fertile zone; the release remains unbuilt. |
| [#2](https://github.com/kermitkiller666/Bibite-GPU/issues/2) | Missing brain, settings, food, collision and attack features | Important and substantially accurate. New native brains now route the Constant input, 33 named inputs and 15 outputs through GPU behavior, with a 12+12 hidden topology. Neuron counts remain fixed, and food/combat/settings do not yet reproduce every stock mechanic. The native tests and an isolated selected-brain visual check passed. Longer-run exit delay, ecological viability, broad gameplay checks and portable packaging remain. |

## Work started on the zone report

The native engine now accepts up to 64 plant zones and samples initial and
regrown food inside their circle, center-biased circle, edge-biased circle,
ring, flat ring or rectangle distributions. Managed code projects the stock
zone settings into the GPU world at startup and polls live settings for
future spawn changes. Plant-only zone weights drive placement and growth;
meat zones no longer contribute to the plant pellet target estimate.

`BibitesGpuWorldTests.exe --food-zones-only` passed on the RTX 4070 Ti, the
managed plugin compiled, and the core test executable passed. The zone test
now verifies both pellets and 32 newly spawned Bibites lie in one of the two
test zones. A limited isolated Unity playtest showed the 3 Islands scenario
opening on a fertile island with Bibites and pellets visible. Existing
pellets are not teleported when a zone is edited; new and respawned food use
the updated zone. Native checkpoint files do not yet store zone definitions
by themselves; the Unity wrapper must reproject the scenario zones on load.

Subsequent source work made disabled hidden links contribute zero while
retaining their FP16 strength for reconnection, inheritance and save/load.
The 0.6.10 native code now writes checkpoint format 11 and reads older GPU
checkpoints; older binaries cannot read format 11. Focused CUDA link,
migration and evolving-brain regressions passed. The 73-node native brain
graph was visually inspected in isolated Unity with the matching managed and
core DLLs; the portable EXE remains the older release.

## Still open from #2

- Exact stock semantics for every sensor and action, and variable neuron-count
  topology evolution. New native Bibites have all 34 sensor slots and 15
  action slots; older Bibites are best-effort imports.
- Exact mouth, stomach, diet and meat/plant digestion formulas. The current
  source has material-separated pellets, stomachs and diet-weighted digestion.
- Exact collision, attack and grab-joint behavior. The current source has
  contact damage, biting, and a one-target grab approximation.
- Broad stock-settings parity. Drag, pellet-size multipliers, diet affinity
  powers and material conversion-efficiency ranges have source projections,
  but many other controls and their in-game effects still need validation.
- Longer Unity playtests, packaging, and larger simulations to check ecology
  and performance. In a five-minute large-map native check, only about 150 of
  512 founders survived and under 10 offspring were born.
- Reliable "Exit to Desktop" after a long run. One three-minute 3 Islands
  test stayed in deferred GPU cleanup for over a minute; a short default-world
  run exited normally. The 0.6.10 plugin now logs shutdown state after five
  seconds, but the underlying intermittent delay is not fixed.
