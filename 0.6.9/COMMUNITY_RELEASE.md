# Community release preparation

This repository contains both Bibites GPU Fork source and a portable public
Windows build, [`Bibites GPU Fork 0.6.9 - PORTABLE.exe`](Bibites%20GPU%20Fork%200.6.9%20-%20PORTABLE.exe?raw=1).
That EXE contains the publicly released The Bibites 0.6.3.1 Windows x64 game,
BepInEx, UnityDoorstop and the GPU mod. It unpacks beside itself on first run.
The source can also be built as a mod-only installation for a separately
obtained compatible game copy. This fork has not been rebased onto upstream
0.6.4 or verified on later releases.

## Build and install

Requirements: Visual Studio 2022 C++ build tools, CUDA Toolkit 12.9, CMake,
.NET SDK capable of building `net472`, BepInEx 5.4.23.5, and a compatible
game installation. The native backend supports the CUDA architectures listed
in `CMakeLists.txt` (currently 75, 86, 89 and 120).

1. In a Visual Studio 2022 x64 developer prompt, run:
   `cmake -S . -B build-native-release -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release`
   followed by `cmake --build build-native-release` and
   `ctest --test-dir build-native-release --output-on-failure`.
2. Build `managed/BibitesGpuFork.csproj` in Release with `GameManagedDir`
   pointing at your own game's `The Bibites_Data/Managed` directory and
   `BepInExCoreDir` pointing at your own BepInEx `core` directory. For example:
   `dotnet build managed/BibitesGpuFork.csproj -c Release -p:GameManagedDir="X:/The Bibites/The Bibites_Data/Managed" -p:BepInExCoreDir="X:/The Bibites/BepInEx/core"`.
3. Install BepInEx in your own game copy. Place the built
   `BibitesGpuFork.dll`, `BibitesGpuFork.Core.dll`, and
   `BibitesGpuNative.dll` together in `BepInEx/plugins/BibitesGpuFork/`.
4. Back up any GPU-native saves before launching. In the game's normal
   Settings menu, open GPU Settings and enable native mode for a new world.

Do not commit an unpacked game tree, local save, configuration, log, screenshot,
or build directory. The portable EXE is the sole intentionally bundled game
artifact in this repository. A mod-only archive should contain only mod DLLs.

## Behaviour and compatibility

- Native GPU worlds use a deliberately compact ecology, not exact stock-game
  biology. In 0.6.9, feeding is bounded by bite speed and stomach capacity,
  pellets shrink over time, and energy is released through digestion. The
  full original mouth, diet, organ, and combat systems remain out of scope.
- Version-3 GPU checkpoints store partially eaten pellets. The new loader
  reads v1/v2 checkpoints and upgrades their food state. Older binaries
  cannot read v3 checkpoints. Keep an independent save backup for rollback.
- Stock CPU worlds retain the original game's biology.
- Native CUDA worlds require a supported NVIDIA GPU; the mod does not provide
  an AMD/Intel backend.

## Publication checklist

- [x] Original game extracted from the supplied public 0.6.3.1 ZIP; the
  portable bundle excludes pre-existing saves, personal configuration and logs.
- [x] Native build and `ctest` pass after the feeding change.
- [x] v3 checkpoint round-trip and synthetic v2 upgrade tested.
- [x] Fresh portable launch reached the menu; BepInEx loaded 0.6.9, selected
  the RTX 4070 Ti, and a default 512/2,048-Bibite GPU world started visibly.
- [ ] Repeat a fresh-world feeding animation and save/reload through the public
  EXE before calling this release fully validated. Headless checkpoint tests
  do not prove the complete UI flow.
- [ ] Choose a source-code license. No license has been selected on the user's
  behalf; until then, publishing the source does not grant community reuse
  rights.
- [x] Confirm that the game ZIP points to the creators' public itch.io 0.6.3.1
  release, not a Patreon-only build; the maintainer has stated that public
  variant distribution is allowed. This is not a claim of official endorsement.
- [x] The 65.8 MB upload is under GitHub's per-file limit; its SHA-256 was
  checked, and the staged file list was reviewed for personal data.

Rollback: restore the previous mod binaries together with a v1/v2 save
backup. Do not open a v3 save with the previous native library.
