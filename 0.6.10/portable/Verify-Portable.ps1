[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Path)
$ErrorActionPreference='Stop'
$taskExe=(Resolve-Path -LiteralPath $Path).Path
# Inspect only: never invoke Main, extract files, or launch the game.
$taskAssembly=[Reflection.Assembly]::LoadFile($taskExe)
$taskType=$taskAssembly.GetType('PortableLauncher',$true)
$taskFlags=[Reflection.BindingFlags]::NonPublic -bor [Reflection.BindingFlags]::Static
function Read-Constant([string]$Name) {
    $taskField=$taskType.GetField($Name,$taskFlags)
    if ($null -eq $taskField) { throw "Missing launcher constant: $Name" }
    return [string]$taskField.GetRawConstantValue()
}
$taskExpectedPayload=Read-Constant 'PayloadSha256'
$taskPayload=$taskAssembly.GetManifestResourceStream('BibitesPayload.zip')
if ($null -eq $taskPayload) { throw 'Embedded payload missing.' }
$taskSha=[Security.Cryptography.SHA256]::Create()
try { $taskActualPayload=[BitConverter]::ToString($taskSha.ComputeHash($taskPayload)).Replace('-','') }
finally { $taskSha.Dispose();$taskPayload.Dispose() }
if ($taskActualPayload -ne $taskExpectedPayload) { throw 'Embedded payload hash mismatch.' }
Add-Type -AssemblyName System.IO.Compression
$taskPayload=$taskAssembly.GetManifestResourceStream('BibitesPayload.zip')
$taskArchive=[IO.Compression.ZipArchive]::new($taskPayload,[IO.Compression.ZipArchiveMode]::Read,$false)
try {
    $taskPlugins=@($taskArchive.Entries | Where-Object { $_.FullName -match '^BepInEx/plugins/.*\.dll$' })
    $taskExpected=@{
        'BepInEx/plugins/BibitesGpuFork/BibitesGpuFork.Core.dll'=(Read-Constant 'CoreSha256')
        'BepInEx/plugins/BibitesGpuFork/BibitesGpuFork.dll'=(Read-Constant 'PluginSha256')
        'BepInEx/plugins/BibitesGpuFork/BibitesGpuNative.dll'=(Read-Constant 'NativeSha256')
    }
    if ($taskPlugins.Count -ne 3) { throw 'Unexpected/duplicate plugin DLLs in payload.' }
    foreach ($taskName in $taskExpected.Keys) {
        $taskEntry=$taskArchive.GetEntry($taskName)
        if ($null -eq $taskEntry) { throw "Missing plugin: $taskName" }
        $taskStream=$taskEntry.Open();$taskSha=[Security.Cryptography.SHA256]::Create()
        try { $taskHash=[BitConverter]::ToString($taskSha.ComputeHash($taskStream)).Replace('-','') }
        finally { $taskSha.Dispose();$taskStream.Dispose() }
        if ($taskHash -ne $taskExpected[$taskName]) { throw "Bundled plugin hash mismatch: $taskName" }
        "$taskName : $taskHash"
    }
    foreach ($taskName in @('The Bibites.exe','UnityPlayer.dll','winhttp.dll','README-PORTABLE.txt','THIRD_PARTY_NOTICES.txt',
        'LICENSES/BepInEx-LICENSE.txt','LICENSES/UnityDoorstop-LICENSE.txt')) {
        if ($null -eq $taskArchive.GetEntry($taskName)) { throw "Missing portable input: $taskName" }
    }
    $taskUnexpected=@($taskArchive.Entries | Where-Object {
        $_.FullName -match '(?i)(^|/)(config|saves|screenshots)/|\.log$|\.cfg$|\.bgfgpu$'
    })
    if ($taskUnexpected.Count) { throw 'Personal/runtime data unexpectedly present in payload.' }
    $taskNames=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($taskEntry in $taskArchive.Entries) {
        $taskRelative=$taskEntry.FullName.Replace('/','\')
        $taskRoot='C:\BibitesPortableValidation\'
        if (-not $taskNames.Add($taskRelative) -or [IO.Path]::IsPathRooted($taskRelative) -or
            $taskRelative.Contains(':') -or -not [IO.Path]::GetFullPath($taskRoot+$taskRelative).StartsWith(
                $taskRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe or duplicate ZIP path.' }
    }
    if ($taskArchive.Entries.Count -gt 10000) { throw 'Too many payload entries.' }
    $taskBytes=($taskArchive.Entries | Measure-Object -Property Length -Sum).Sum
    if ($taskBytes -gt 512MB) { throw 'Payload exceeds launcher extraction ceiling.' }
    "Embedded payload SHA256: $taskActualPayload"
    "Archive entries: $($taskArchive.Entries.Count); uncompressed bytes: $taskBytes"
} finally { $taskArchive.Dispose() }

Get-Item -LiteralPath $taskExe | Select-Object FullName,Length | Format-List
Get-FileHash -LiteralPath $taskExe -Algorithm SHA256 | Format-List
'Portable verification passed; no game was launched.'
