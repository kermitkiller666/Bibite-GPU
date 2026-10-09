[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$CleanGameDirectory,
    [Parameter(Mandatory=$true)][string]$NativeLibraryPath,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$ManagedOutputDirectory = (Join-Path $PSScriptRoot '..\managed\bin\Release')
)
$ErrorActionPreference = 'Stop'
$taskBase = (Resolve-Path -LiteralPath $CleanGameDirectory).Path
$taskNative = (Resolve-Path -LiteralPath $NativeLibraryPath).Path
$taskManaged = (Resolve-Path -LiteralPath $ManagedOutputDirectory).Path
$taskOutput = [IO.Path]::GetFullPath($OutputDirectory)
$taskPayload = Join-Path $taskOutput 'BibitesPayload.zip'
$taskLauncher = Join-Path $taskOutput 'PortableLauncher.generated.cs'
$taskExe = Join-Path $taskOutput 'Bibites GPU Fork 0.6.10 - PREVIEW.exe'
$taskCsc = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$taskInputs = @{
    'BepInEx/plugins/BibitesGpuFork/BibitesGpuFork.Core.dll' = (Join-Path $taskManaged 'BibitesGpuFork.Core.dll')
    'BepInEx/plugins/BibitesGpuFork/BibitesGpuFork.dll' = (Join-Path $taskManaged 'BibitesGpuFork.dll')
    'BepInEx/plugins/BibitesGpuFork/BibitesGpuNative.dll' = $taskNative
    'README-PORTABLE.txt' = (Join-Path $PSScriptRoot 'README-PORTABLE.txt')
    'THIRD_PARTY_NOTICES.txt' = (Join-Path $PSScriptRoot 'THIRD_PARTY_NOTICES.txt')
}
foreach ($taskPath in @($taskPayload, $taskLauncher, $taskExe)) {
    if (Test-Path -LiteralPath $taskPath) { throw "Existing output is never overwritten: $taskPath" }
}
if ($taskOutput -eq $taskBase -or $taskOutput.StartsWith(
        $taskBase.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Output must be outside the clean input game folder.'
}
foreach ($taskPath in @($taskInputs.Values) + @($taskCsc,
        (Join-Path $taskBase 'The Bibites.exe'), (Join-Path $taskBase 'UnityPlayer.dll'),
        (Join-Path $taskBase 'winhttp.dll'),
        (Join-Path $taskBase 'LICENSES\BepInEx-LICENSE.txt'),
        (Join-Path $taskBase 'LICENSES\UnityDoorstop-LICENSE.txt'))) {
    if (-not (Test-Path -LiteralPath $taskPath -PathType Leaf)) { throw "Missing input: $taskPath" }
}
[IO.Directory]::CreateDirectory($taskOutput) | Out-Null
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($taskBase, $taskPayload,
    [IO.Compression.CompressionLevel]::Optimal, $false)
$taskArchive = [IO.Compression.ZipFile]::Open($taskPayload, [IO.Compression.ZipArchiveMode]::Update)
try {
    foreach ($taskName in $taskInputs.Keys) {
        $taskOldEntry = $taskArchive.GetEntry($taskName)
        if ($null -ne $taskOldEntry) { $taskOldEntry.Delete() }
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($taskArchive,
            $taskInputs[$taskName], $taskName, [IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
    $taskPersonal = @($taskArchive.Entries | Where-Object {
        $_.FullName -match '(?i)(^|/)(config|saves|screenshots)/|\.log$|\.cfg$|\.bgfgpu$'
    })
    if ($taskPersonal.Count) { throw 'Input contains runtime/personal files; use a clean staging game.' }
    $taskPlugins = @($taskArchive.Entries | Where-Object { $_.FullName -match '(?i)^BepInEx/plugins/.*\.dll$' })
    if ($taskPlugins.Count -ne 3) { throw 'Only the three canonical fork DLLs may be bundled.' }
    $taskBytes = ($taskArchive.Entries | Measure-Object -Property Length -Sum).Sum
    if ($taskArchive.Entries.Count -gt 10000 -or $taskBytes -gt 512MB) {
        throw 'Payload exceeds the launcher file/size ceiling.'
    }
} finally { $taskArchive.Dispose() }
$taskSource = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'PortableLauncher.cs'))
$taskHashes = @{
    '__PAYLOAD_SHA256__' = (Get-FileHash -LiteralPath $taskPayload -Algorithm SHA256).Hash
    '__CORE_SHA256__' = (Get-FileHash -LiteralPath $taskInputs['BepInEx/plugins/BibitesGpuFork/BibitesGpuFork.Core.dll'] -Algorithm SHA256).Hash
    '__PLUGIN_SHA256__' = (Get-FileHash -LiteralPath $taskInputs['BepInEx/plugins/BibitesGpuFork/BibitesGpuFork.dll'] -Algorithm SHA256).Hash
    '__NATIVE_SHA256__' = (Get-FileHash -LiteralPath $taskNative -Algorithm SHA256).Hash
}
foreach ($taskMarker in $taskHashes.Keys) { $taskSource = $taskSource.Replace($taskMarker, $taskHashes[$taskMarker]) }
# Generated compiler input is a build product; the tracked template stays unchanged.
[IO.File]::WriteAllText($taskLauncher, $taskSource, [Text.UTF8Encoding]::new($false))
& $taskCsc /nologo /target:winexe /platform:x64 /optimize+ "/out:$taskExe" `
    /reference:System.Drawing.dll /reference:System.Windows.Forms.dll `
    /reference:System.IO.Compression.dll /reference:System.IO.Compression.FileSystem.dll `
    "/resource:$taskPayload,BibitesPayload.zip" $taskLauncher
if ($LASTEXITCODE -ne 0) { throw "Portable compiler failed: $LASTEXITCODE" }
Get-FileHash -LiteralPath $taskExe -Algorithm SHA256
Get-Item -LiteralPath $taskExe | Select-Object FullName, Length
