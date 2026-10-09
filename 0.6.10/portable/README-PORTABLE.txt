Bibites GPU Fork 0.6.10 PREVIEW - portable single-file download

This unofficial GPU variant contains the public The Bibites 0.6.3.1 Windows
x64 game. 0.6.10 is the fork version, not an upstream 0.6.10 or 0.6.4 game.
It is an experimental preview, not a stable or stock-parity release.

Double-click in a writable folder. The launcher verifies and extracts the
game into a versioned preview-data folder beside itself. It needs no installer
or administrator access. Keep extracted folders and back up both the .zip
and .zip.bgfgpu save files. Moving the EXE alone does not transfer user data.
The launcher is unsigned; download only from the project's GitHub repository.

Settings > GPU Settings selects the NVIDIA CUDA GPU, population cap, graphics
FPS, on-screen texture budget (0-2048; default 512), Exact/Extreme profile and
fused/sparse/dense kernel for the next world. Graphics FPS applies live.
Native checkpoints retain their kernel choice. AMD/Intel GPU simulation is
not implemented. The normal CUDA path does not require Tensor Cores.

New brains have 34 inputs, 12+12 hidden neurons and 15 outputs. FP16 weights
and sparse connection masks evolve; disabled links have effective weight zero
and retain their genetic weight for reactivation. Accumulation/physics are FP32.
New food caps respect stock biomass and the configured GPU pellet ceiling.
Meat, diets, collisions, attacks, growth and other actions are approximations.

Optimizations include segmented local food overflow, parallel dense-food
sensing, shared-memory contact tiles, compact live work lists, cached invariant
math and CUDA Graphs. The fused kernel stays default. Tensor Core evaluation is
an opt-in CLI prototype, not consistently faster. Requested warp (up to 10000x)
is not a guaranteed achieved speed; kernel share is not GPU utilization.

GPU render captures and disk writers own independent snapshots. Save capture
still briefly pauses the simulation worker. This preview writes checkpoint
FORMAT 12; older binaries cannot read it. Retain older builds and save new
worlds under new names. Three companion DLL hashes are checked on extraction
and reuse to prevent a stale library from silently joining the new build.

The October 9 review bounded two graphics handoff waits to 15 seconds. On
timeout, uncertain callback/world/buffer ownership is quarantined until exit,
not forcibly freed or reset. Restart after an interop fault. This does not fix
a driver call that itself never returns and is not an in-game shutdown guarantee.

Fresh core, managed and native checks passed on RTX 4070 Ti. Headless tests
include brain parity, food/contact handling, save/load/resume and render overlap.
Final Unity GUI, save-wrapper, menu-exit and FPS validation is outstanding.
Original close-up sprite parts are still CPU-managed, not fully GPU instanced.
Ecology remains experimental: a five-minute regression retained 152 of 512
founders with 8 births and 368 starvation deaths. Uniform dense-food benchmarks
can be slower even though clustered-food queries improved.

Requirements: Windows x64, .NET Framework 4.7.2 or newer, supported NVIDIA CUDA
GPU/driver and about 181 MiB for extracted files, plus space for saves.
No existing personal saves/configuration/logs/screenshots are included.
For bug reports attach BepInEx/LogOutput.log and Unity Player.log after reviewing
them for personal information. See THIRD_PARTY_NOTICES.txt for attribution.
