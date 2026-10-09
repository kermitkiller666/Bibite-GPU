# 0.6.10 preview bundle

Build: October 9, 2026. Single-file Windows x64 download:
[Bibites GPU Fork 0.6.10 - PREVIEW.exe](Bibites%20GPU%20Fork%200.6.10%20-%20PREVIEW.exe?raw=1).

The EXE is **72,923,648 bytes** (about 73 MB). It embeds a ZIP with 243 entries
and 189,653,080 uncompressed bytes (about 181 MiB), extracted beside the EXE
into `Bibites GPU Fork 0.6.10 - preview-data-2f2ffa4728db`.
It is unsigned and requires .NET Framework 4.7.2 or newer. It is a single-file
download, not a memory-only game: Unity requires the extracted folder.

## SHA-256

| Component | SHA-256 |
| --- | --- |
| Portable EXE | `AFA1C4F6587BE047C06B99C574B525FD93D316EA05FD0B7ADFEB58679808F732` |
| Embedded ZIP | `2F2FFA4728DB7AB5DC2AD118ED430BDD3A9A7484634AADDAD46910A8B9D33D1F` |
| BibitesGpuFork.Core.dll | `B7C27C9A3FB69401045AB073992B6D098936486DEA2808C9273AB4D8A1107BD2` |
| BibitesGpuFork.dll | `3BA2209FF6EFF2BE33B0177184A44FA990BDE7D58807E236D974A62A3F9B09C4` |
| BibitesGpuNative.dll | `1BFE26B151EA12D470DA97C10A2133F2B3C498D6E1A98F5321BE6B26C37278F7` |

All three fork DLLs are in the canonical `BepInEx/plugins/BibitesGpuFork/`
directory. The launcher validates the ZIP hash during extraction and each fork
DLL hash before launch/reuse. These hashes identify the published artifact;
fresh builds may differ due to ZIP/build timestamps and are not byte-reproducible.

## Provenance and contents

The base is the creator's public **The Bibites 0.6.3.1 Windows x64** distribution
obtained from [itch.io](https://thebibites.itch.io/the-bibites). It is not a
Patreon-only build, an upstream 0.6.10 release, or a rebase onto 0.6.4. Original
game ownership remains with its creators; publication does not give this project
ownership of their game. See the included third-party notices and licenses.

The payload includes the public game, Unity runtime, BepInEx 5.4.23.5,
UnityDoorstop 4.5.0 and the matching fork libraries. Loader license text remains
in `LICENSES/`, and [THIRD_PARTY_NOTICES.txt](portable/THIRD_PARTY_NOTICES.txt)
links their sources. No general license has been selected for the fork code.

The build used a clean staging game, not a running/extracted user's game.
Existing personal configuration, saves, logs, screenshots and test outputs are
excluded. Previous local PARITY, SAVE FIX and OPTIMIZED previews are unchanged.

## Validation boundary

[October 9 review](docs/code-check-2026-10-09.md) documents fresh core/native
tests and the 15-second managed graphics-handoff deadline. This package was
verified without executing the game. Final Unity menu/save-wrapper/FPS/shutdown
verification remains outstanding; do not infer a graphical playtest from the
headless render-overlap test. Retain backups: new checkpoints use format 12
and cannot be read by older fork binaries.
