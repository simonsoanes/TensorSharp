param(
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Release',
    [string[]]$PackageSource = @(),
    [switch]$Run
)

# Build and verify the complete app before offering its executable. An older
# binary can still exist after a failed build, so never launch it on failure.
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$projectPath = Join-Path $repoRoot 'TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj'
$outputPath = Join-Path $repoRoot "TensorAgent/src/TensorAgent.Maui/bin/$Configuration/net10.0-windows10.0.19041.0/win-x64"
$binaryPath = Join-Path $outputPath 'TensorAgent.Maui.exe'
$runningApp = Get-Process -Name TensorAgent.Maui -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $binaryPath }
if ($runningApp) { throw 'Close this TensorAgent instance before rebuilding it.' }

$buildArgs = @('build', $projectPath, '-f', 'net10.0-windows10.0.19041.0', '-r', 'win-x64', '-c', $Configuration, '-m:1', '--nologo')
if ($PackageSource.Count) {
    $restoreArgs = @('restore', $projectPath, '-r', 'win-x64', '--nologo')
    foreach ($source in $PackageSource) { $restoreArgs += @('--source', $source) }
    & dotnet @restoreArgs
    if ($LASTEXITCODE -ne 0) { throw 'TensorAgent package restore failed; the existing executable will not be launched.' }
    $buildArgs += '--no-restore'
}
& dotnet @buildArgs
if ($LASTEXITCODE -ne 0) { throw "TensorAgent build failed. The existing executable was not updated: $binaryPath" }
if (-not (Test-Path -LiteralPath $binaryPath)) { throw "Build did not produce the expected executable: $binaryPath" }

$corePath = Join-Path $outputPath 'TensorAgent.Core.dll'
# Read from memory so invoking this script in an interactive shell does not lock
# the DLL and prevent the next build from replacing it.
$assembly = [Reflection.Assembly]::Load([IO.File]::ReadAllBytes($corePath))
$resources = @{
    'TensorAgent.Core.WebUi.tensoragent.js' = 'TensorAgent/src/TensorAgent.Core/WebUi/tensoragent.js'
    'TensorAgent.Core.WebUi.mask-editor.js' = 'TensorSharp.Chat/WebUi/mask-editor.js'
    'TensorAgent.Core.WebUi.mask-editor.css' = 'TensorSharp.Chat/WebUi/mask-editor.css'
}
foreach ($resourceName in $resources.Keys) {
    $stream = $assembly.GetManifestResourceStream($resourceName)
    if (-not $stream) { throw "The built app is missing $resourceName." }
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { $embeddedHash = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '') }
    finally { $stream.Dispose(); $hasher.Dispose() }
    $sourceHash = (Get-FileHash -LiteralPath (Join-Path $repoRoot $resources[$resourceName]) -Algorithm SHA256).Hash
    if ($embeddedHash -ne $sourceHash) { throw "The built app contains an outdated $resourceName." }
}
$sourcePageHash = (Get-FileHash -LiteralPath (Join-Path $repoRoot 'TensorAgent/src/TensorAgent.Maui/wwwroot/index.html') -Algorithm SHA256).Hash
$bundledPageHash = (Get-FileHash -LiteralPath (Join-Path $outputPath 'webui/index.html') -Algorithm SHA256).Hash
if ($sourcePageHash -ne $bundledPageHash) { throw 'The built app contains an outdated webui/index.html.' }

Write-Output "Verified TensorAgent executable: $binaryPath"
if ($Run) { & $binaryPath }
