# Requires PowerShell 7, Python 3, and WiX 6.0.2 on Windows.
# Install the pinned tool without modifying a developer's global tool set:
#   dotnet tool install wix --version 6.0.2 --tool-path artifacts/tools/wix
#   ./eng/package-tensoragent-windows.ps1 -PublishDirectory publish/desktop `
#     -Version 2026.10.03 -Variant win-x64-cpu -OutputDirectory artifacts `
#     -WixPath artifacts/tools/wix/wix.exe
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $PublishDirectory,
    [Parameter(Mandatory)][string] $Version,
    [Parameter(Mandatory)][ValidateSet("win-x64-cpu", "win-x64-cuda")][string] $Variant,
    [Parameter(Mandatory)][string] $OutputDirectory,
    [string] $WixPath = "wix",
    [string] $PythonPath = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (-not $IsWindows) { throw "Windows MSI packaging and payload extraction require Windows." }

$publish = (Resolve-Path -LiteralPath $PublishDirectory).Path
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
$publishPrefix = $publish.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if ($output.TrimEnd([System.IO.Path]::DirectorySeparatorChar) -eq $publish.TrimEnd([System.IO.Path]::DirectorySeparatorChar) -or
    $output.StartsWith($publishPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Package output must be outside the publish directory."
}
$work = Join-Path $output "package-tensoragent-windows/$Variant"
New-Item -ItemType Directory -Force $work | Out-Null
$source = Join-Path $work "TensorAgent.wxs"
$manifestPath = Join-Path $work "payload.json"

& $PythonPath (Join-Path $PSScriptRoot "generate-tensoragent-wix.py") `
    --publish-directory $publish --version $Version --variant $Variant `
    --output $source --manifest $manifestPath
if ($LASTEXITCODE -ne 0) { throw "TensorAgent publish payload validation failed." }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$prefix = "tensoragent-desktop-$Version-$Variant"
$zipPath = Join-Path $output "$prefix.zip"
$msiPath = Join-Path $output "$prefix.msi"

$wixVersion = & $WixPath --version
if ($LASTEXITCODE -ne 0 -or "$wixVersion" -notmatch '^6\.0\.2(?:\+|$)') {
    throw "Install the pinned WiX 6.0.2 tool before packaging. Found: $wixVersion"
}
# WiX runs its normal MSI consistency validation. Keep source/debug databases
# under the work directory, outside the release asset glob.
& $WixPath build $source -arch x64 -o $msiPath -pdb (Join-Path $work "TensorAgent.wixpdb")
if ($LASTEXITCODE -ne 0) { throw "WiX failed to build/validate TensorAgent Desktop MSI." }

Add-Type -AssemblyName System.IO.Compression.FileSystem
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath }
# ZipFile includes hidden payload files and does not have Compress-Archive's
# documented 2 GB per-file limit. The executable remains at the archive root.
[System.IO.Compression.ZipFile]::CreateFromDirectory($publish, $zipPath, [System.IO.Compression.CompressionLevel]::Optimal, $false)
$archive = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    $entries = @{}
    foreach ($entry in $archive.Entries) {
        if ($entry.Name -eq "") { continue }
        $name = $entry.FullName.Replace('\', '/')
        if ($entries.ContainsKey($name)) { throw "Duplicate ZIP entry: $name" }
        $entries[$name] = $entry
    }
    if ($entries.Count -ne $manifest.files.Count) { throw "ZIP payload file count differs from validated publish output." }
    foreach ($file in $manifest.files) {
        if (-not $entries.ContainsKey($file.path)) { throw "ZIP is missing $($file.path)." }
        $stream = $entries[$file.path].Open()
        $algorithm = [System.Security.Cryptography.SHA256]::Create()
        try { $hash = [System.Convert]::ToHexString($algorithm.ComputeHash($stream)).ToLowerInvariant() }
        finally { $algorithm.Dispose(); $stream.Dispose() }
        if ($hash -ne $file.sha256) { throw "ZIP payload hash mismatch: $($file.path)" }
    }
} finally { $archive.Dispose() }

# An administrative extraction checks the built MSI's cabinet and paths without
# registering or launching the app on the runner. It is not a GUI/install test.
$extracted = Join-Path $work "msi-extracted"
if (Test-Path -LiteralPath $extracted) { Remove-Item -LiteralPath $extracted -Recurse -Force }
New-Item -ItemType Directory -Force $extracted | Out-Null
$extractLog = Join-Path $work "msi-extract.log"
$arguments = @("/a", ('"' + $msiPath + '"'), ('TARGETDIR="' + $extracted + '"'), "/qn", "/norestart", "/l*v", ('"' + $extractLog + '"'))
$process = Start-Process -FilePath "msiexec.exe" -ArgumentList $arguments -Wait -PassThru
if ($process.ExitCode -ne 0) { throw "MSI administrative extraction failed ($($process.ExitCode)); see $extractLog." }
$executables = @(Get-ChildItem -LiteralPath $extracted -Filter TensorAgent.Maui.exe -Recurse -File)
if ($executables.Count -ne 1) { throw "MSI must contain exactly one TensorAgent.Maui.exe." }
$extractedPayload = $executables[0].Directory.FullName
$extractedFiles = @(Get-ChildItem -LiteralPath $extractedPayload -Recurse -File -Force)
if ($extractedFiles.Count -ne $manifest.files.Count) { throw "MSI payload file count differs from validated publish output." }
foreach ($file in $manifest.files) {
    $path = Join-Path $extractedPayload $file.path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "MSI is missing $($file.path)." }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) {
        throw "MSI payload hash mismatch: $($file.path)"
    }
}

$checksums = foreach ($path in @($zipPath, $msiPath)) {
    if ((Get-Item -LiteralPath $path).Length -eq 0) { throw "Empty release package: $path" }
    "$((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant())  $([System.IO.Path]::GetFileName($path))"
}
$checksums | Set-Content -LiteralPath (Join-Path $output "SHA256SUMS-$prefix.txt") -Encoding utf8NoBOM
@(
    "Release: $Version; MSI version: $($manifest.msi_version); variant: $Variant; WiX: $wixVersion",
    "Validated $($manifest.files.Count) self-contained/native/Web UI/skill payload files.",
    "Verified ZIP entries and MSI administrative extraction against each payload SHA-256.",
    "Not exercised: user installation, upgrade/variant switching, uninstall, GUI launch, model inference, or CUDA devices."
) | Set-Content -LiteralPath (Join-Path $work "VALIDATION.txt") -Encoding utf8NoBOM
Write-Host "Created $zipPath and $msiPath; verified $($manifest.files.Count) payload files in both packages."
