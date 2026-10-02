# Community release preparation

This repository contains the Bibites GPU Fork mod source only. The local
`READY TO RUN` archive contains the original game and must **not** be pushed
to this repository or attached to a public release. Users need their own
compatible Windows x64 copy of The Bibites 0.6.3.1. This fork has not been
rebased onto upstream 0.6.4 or verified on later releases.

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

No original game DLL, asset, launcher, save, BepInEx binary, or local build
directory belongs in the Git repository or a mod-only release archive.

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

- [x] Source-only ignore rules added; original game and local build outputs
  excluded from the intended GitHub repository.
- [x] Native build and `ctest` pass after the feeding change.
- [x] v3 checkpoint round-trip and synthetic v2 upgrade tested.
- [ ] Test a fresh graphical world, settings UI, feeding animation, save and
  reload in the actual game before tagging a public release.
- [ ] Choose a source-code license. No license has been selected on the user's
  behalf; until then, publishing the source does not grant community reuse
  rights.
- [ ] Review upstream game/mod distribution terms before attaching any
  binaries. A source-only repository does not require shipping the game.
- [ ] Create a GitHub remote and push only after reviewing `git status` and
  `git ls-files` for third-party files or personal data.

Rollback: restore the previous mod binaries together with a v1/v2 save
backup. Do not open a v3 save with the previous native library.
