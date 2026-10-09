# Bibites GPU Fork

## Latest: 0.6.10 preview (October 9)

**[Download the complete 0.6.10 PREVIEW.exe (about 73 MB)](0.6.10/Bibites%20GPU%20Fork%200.6.10%20-%20PREVIEW.exe?raw=1)**

The new [source and setup notes](0.6.10/README.md) include 34-input,
12+12-hidden, 15-output evolving brains, food-cap/zone fixes, meat/diet/action
approximations, local dense-food queries, tiled contacts, specialized neural
kernels, owned render snapshots and asynchronous compact checkpoints.
The standard fused CUDA kernel remains the default. Windows x64 and a supported
NVIDIA CUDA GPU/driver are required; AMD acceleration remains deferred.

The October 9 code review bounded two graphics-callback waits to 15 seconds.
Fresh core/native regression tests and the managed build passed on RTX 4070 Ti.
[Review results and known limits](0.6.10/docs/code-check-2026-10-09.md) distinguish
these checks from an in-game test: final Unity GUI/save-wrapper/FPS/shutdown
validation is still outstanding, ecology is experimental, and original sprite
parts are not fully GPU-instanced. This is a **preview**, not stock-game parity.

Back up both save files and keep older builds: new checkpoints use **format 12**,
which older binaries cannot read. The download uses the public 0.6.3.1 game,
not a Patreon-only release or a 0.6.4 rebase. See [provenance/checksums](0.6.10/BUNDLE_MANIFEST.md).

Portable EXE SHA-256:
`AFA1C4F6587BE047C06B99C574B525FD93D316EA05FD0B7ADFEB58679808F732`

## Previous public 0.6.9 build

An **unofficial CUDA-accelerated variant of The Bibites** for Windows x64. This
repository contains the GPU-fork source and a portable **0.6.9** game build.
It is built on the creators' publicly released **The Bibites 0.6.3.1** from
[itch.io](https://thebibites.itch.io/the-bibites). The `0.6.9` number is this
fork's version; it is **not** an upstream 0.6.9 release or a rebase of 0.6.4.

The goal is to make large, fast-running evolutionary worlds practical while
keeping the original graphical application, menus, camera, and inspection UI.
The GPU-native simulation is an experimental approximation of the game's
ecology, **not a bit-for-bit or feature-complete replacement** for the stock
simulation.

## Download and play

**[Download Bibites GPU Fork 0.6.9 - PORTABLE.exe](0.6.9/Bibites%20GPU%20Fork%200.6.9%20-%20PORTABLE.exe?raw=1)**

The download is a single, roughly 66 MB Windows EXE containing the public
0.6.3.1 game, the GPU mod, BepInEx, and UnityDoorstop. You do **not** need to
separately install the base game for this build.

1. Save the EXE in a folder you can write to (for example, a new folder in
   Documents), and double-click it.
2. On first launch it verifies and unpacks about 161 MiB into a versioned
   `Bibites GPU Fork 0.6.9 - data-*` folder beside the EXE. Later launches reuse
   that folder. Keep the folder if you want to retain game-local files.
3. In the game, open **Settings → GPU Settings** to choose your NVIDIA GPU and
   set the population cap, starting population, graphics FPS, texture budget,
   and Exact/Extreme profile. Start a **new** simulation for the selected GPU
   and most world settings to take effect.

This is a one-file **download**, not a memory-only executable: Unity needs the
extracted folder to run. The launcher does not need an installer or admin
access. Moving only the EXE to another PC does not transfer saves or settings.
The launcher is not code-signed; only download it from this repository and
check the published SHA-256 below if you want to verify the file.

| Requirement | Details |
| --- | --- |
| Operating system | Windows x64 |
| Runtime | .NET Framework 4.7.2 or newer for the portable launcher |
| GPU-native world | Supported NVIDIA CUDA GPU and working driver |
| Other GPUs | The original CPU game remains available; AMD/Intel GPU acceleration is not implemented |
| Disk space | About 66 MB for the download, plus about 161 MiB unpacked and room for saves |

Portable EXE SHA-256:

```text
80E14864567F6ED48382A50F5A162233800F3EA19B831977BE9676F557B55BCC
```

## What this build adds

- A GPU-native world with its simulation state and most stepping work held on
  one CUDA device, with the original Unity interface still visible.
- A selectable time-warp ladder:
  `1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000x`.
  These are **requested** speeds, not guaranteed achieved speeds; the HUD
  reports the measured result. The largest settings are intended for small
  worlds, around 256 Bibites.
- Per-world population settings up to a 500,000-Bibite hard ceiling. A smaller
  selected cap allocates for that smaller world; a high cap does not imply that
  your GPU can simulate it in real time.
- Independent graphics-FPS and close-up textured-Bibite limits. Distant
  Bibites use a lightweight GPU silhouette rather than all receiving the full
  Unity procedural sprite cost.
- A CPU-side inspector and species/lineage browser backed by native GPU-world
  state, plus the game's normal save/load interface for native worlds.
- In 0.6.9, bounded bites and stomach capacity: pellets shrink as they are
  eaten, and digestion releases energy over time rather than instantly adding
  an entire pellet's energy.

## Saves, settings, and controls

GPU-native saves use **two files**: a normal `<name>.zip` wrapper and a
`<name>.zip.bgfgpu` checkpoint. Copy or back up both together. Version-3
checkpoints from 0.6.9 preserve partially eaten pellets and stomach state;
older fork binaries cannot read those saves. Back up important worlds before
switching versions. Some game and Unity settings may also live in your Windows
user profile, so the extracted folder alone is not a complete PC migration.

| Control | Action |
| --- | --- |
| Ctrl + Page Up / Page Down | Change requested time warp |
| Page Up / Page Down | Original camera-speed controls |
| Space | Pause or resume |
| F5 / F9 | Quick-save / quick-load |
| 1–5 | Switch selected Bibite inspector tabs |

The selected GPU, cap, starting population, native mode, Exact/Extreme profile,
and texture limit apply to the **next** world. Graphics FPS applies live.
Dynamic Settings for food density, fertility/regrowth, and pellet energy update
the running GPU-native world. Loading an existing stock save uses CPU
compatibility for that session; it does not convert every stock entity to the
compact GPU model.

## Important differences from the original game

This is a performance-focused fork, not the original game's biology moved
unchanged onto CUDA. Native worlds do not fully implement stock organs, diet,
meat, attacks, grabbing, zones, towers, structural NEAT evolution, or the
complete automatic species tree. The lineage browser groups living lineages;
it is not the original historical family tree. Above 8,192 living Bibites,
many charts, lineage distributions, and picking use a representative sample;
the headline population is exact. Use a stock CPU world when those original
mechanics are more important than acceleration.

The engine uses FP32 for positions, biology, neuron state, and accumulation;
FP16 stores inherited neural weights. A requested 10000x does not mean the
machine will actually deliver 10000x. Achieved speed depends on population,
brain complexity, spatial clustering, world size, graphics settings, and other
GPU work. See the [technical notes](0.6.9/README.md) for measured tests and
their limits.

## Source, build, and troubleshooting

- [0.6.9 source and technical notes](0.6.9/README.md)
- [Portable bundle provenance and checksums](0.6.9/BUNDLE_MANIFEST.md)
- [Source build and mod-only installation](0.6.9/COMMUNITY_RELEASE.md)
- [Detailed configuration and save notes](0.6.9/INSTALL.txt)
- [Engine design](0.6.9/GPU_ENGINE_DESIGN.md)
- [Dated UI and stability validation](0.6.9/docs/gui-stability-2026-10-01.md)

If the EXE does not launch, check Windows x64, .NET Framework 4.7.2+, free disk
space, and write access to its folder. If the game opens but GPU mode is
unavailable, inspect **Settings → GPU Settings** and
`Bibites GPU Fork 0.6.9 - data-*/BepInEx/LogOutput.log`. Do not delete the
extracted folder when troubleshooting if it may contain saves. File an issue
with the fork version, Windows/GPU/driver details, the steps to reproduce, and
relevant log lines after removing personal information.

Contributions and reproducible bug reports are welcome. The source code is
publicly readable but **no general source-code license has been selected**;
do not assume that source publication grants redistribution or relicensing
rights. The included game remains the property of [The Bibites](https://thebibites.itch.io/the-bibites).
BepInEx and UnityDoorstop retain their own licenses, included in the portable
bundle's `LICENSES` folder. This project is unofficial and not endorsed by
the game's creators.
