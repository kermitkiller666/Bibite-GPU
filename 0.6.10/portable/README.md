# Build and verify the portable preview

The checked-in EXE is the complete single-file **download**. It extracts a game
folder when launched. The build script never changes its input game or overwrites
an existing output. It does not download third-party assets or grant distribution
rights; supply a legitimately obtained, clean public 0.6.3.1 staging copy with
BepInEx, UnityDoorstop and their license notices already installed.

Do not use a currently running game folder: configuration, saves, logs and
screenshots must not be bundled. The script rejects known runtime/personal files
and duplicate/extra plugin DLLs. Keep only the three canonical fork DLLs in the
clean game's `BepInEx/plugins/BibitesGpuFork/` directory. Output must be outside
the input game directory, and the output filenames must not already exist.

First build the matching managed/Core and native libraries as described in
[the source README](../README.md). Then, from the repository root:

```powershell
& 0.6.10/portable/Build-Portable.ps1 `
  -CleanGameDirectory 'C:/staging/clean-bibites-0.6.3.1' `
  -NativeLibraryPath 'build/0.6.10/Release/BibitesGpuNative.dll' `
  -OutputDirectory 'build/portable-preview'
& 0.6.10/portable/Verify-Portable.ps1 `
  -Path 'build/portable-preview/Bibites GPU Fork 0.6.10 - PREVIEW.exe'
```

The default managed input is `0.6.10/managed/bin/Release`; override it with
`-ManagedOutputDirectory` if needed. The script uses Windows' x64 .NET Framework
compiler and produces the embedded ZIP, generated launcher compiler input and
EXE as build outputs. The tracked launcher template stays unchanged. Compression
and build timestamps mean a rebuild is not guaranteed to have the published hash.

Verification reads resources without invoking `Main`, extracting or launching
the game. It checks embedded SHA-256, canonical DLL hashes, required files and
licenses, ZIP path/duplicate safety, size limits and known personal-data patterns.
This does not replace an actual authorized in-game smoke test. The launcher also
rejects an existing extracted folder with mismatched companion DLLs, leaving it
untouched so saves are not lost.
