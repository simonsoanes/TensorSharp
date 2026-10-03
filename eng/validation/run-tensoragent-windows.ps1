param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$Weights,
    [string]$Projector = '',
    [string]$Backend = 'ggml_cuda',
    [string]$Python = 'python',
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Debug',
    [int]$Context = 8192,
    [switch]$Tools
)

# Build the Windows head first. Debug exercises its real WebView; both use
# isolated data directories; downloaded catalog models and user data are untouched.
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$evidenceRoot = [IO.Path]::GetFullPath($Root)
$allowed = @('docs/validation', 'artifacts') | Where-Object {
    $evidenceRoot.StartsWith(([IO.Path]::GetFullPath((Join-Path $repo $_)) + [IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)
}
if (-not $allowed) { throw 'Root must be inside repository docs/validation/ or artifacts/.' }
if (Test-Path -LiteralPath $evidenceRoot) { throw 'Use a fresh evidence directory for each run.' }
$weightsPath = (Resolve-Path -LiteralPath $Weights).Path
$projectorPath = if ($Projector) { (Resolve-Path -LiteralPath $Projector).Path } else { '' }
$exe = Join-Path $repo "TensorAgent/src/TensorAgent.Maui/bin/$Configuration/net10.0-windows10.0.19041.0/win-x64/TensorAgent.Maui.exe"
if (-not (Test-Path -LiteralPath $exe)) { throw "Build the $Configuration Windows app first: $exe" }
New-Item -ItemType Directory -Path (Join-Path $evidenceRoot 'Data') -Force | Out-Null
@{ selectedModelId = $null; contextLength = $Context; maxTokens = 512; kvCacheDtype = 'q8_0';
   speculativeDecoding = $false; toolTimeoutSeconds = 300 } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidenceRoot 'Data/settings.json') -Encoding UTF8

$launchEnvironment = @{
    TENSORAGENT_VALIDATION_ROOT = $evidenceRoot
    TENSORAGENT_VALIDATION_WEIGHTS = $weightsPath
    TENSORAGENT_VALIDATION_MMPROJ = $projectorPath
    TENSORAGENT_UI_CHECK = '1'
    TENSORAGENT_DEMO_PROMPT = 'Write a factual explanation of the water cycle in at least 600 words.'
    TENSORAGENT_NAV_CHECK = '1'
    TENSORAGENT_NAV_SECONDS = '3'
    MAX_CONTEXT = [string]$Context
    KV_CACHE_DTYPE = 'q8_0'
}
$oldEnvironment = @{}
$app = $null
$clock = [Diagnostics.Stopwatch]::StartNew()
try {
    foreach ($name in $launchEnvironment.Keys) {
        $oldEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $launchEnvironment[$name], 'Process')
    }
    try {
        $app = Start-Process -FilePath $exe -WorkingDirectory (Split-Path $exe) -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $evidenceRoot 'stdout.log') -RedirectStandardError (Join-Path $evidenceRoot 'stderr.log')
    } finally {
        foreach ($name in $oldEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $oldEnvironment[$name], 'Process') }
    }
    $app.Id | Set-Content -LiteralPath (Join-Path $evidenceRoot 'pid.txt')
    $connectionPath = Join-Path $evidenceRoot 'connection.json'
    while (-not (Test-Path -LiteralPath $connectionPath)) {
        if ($app.HasExited -or $clock.Elapsed.TotalSeconds -gt 60) { throw 'App did not publish its connection within 60 seconds; inspect stdout/stderr.' }
        Start-Sleep -Milliseconds 200
    }
    $startupSeconds = $clock.Elapsed.TotalSeconds
    $connection = Get-Content -LiteralPath $connectionPath -Raw | ConvertFrom-Json
    $session = New-Object Microsoft.PowerShell.Commands.WebRequestSession
    $cookieParts = $connection.cookie -split '=', 2
    $session.Cookies.Add([Uri]$connection.baseUrl, (New-Object Net.Cookie($cookieParts[0], $cookieParts[1], '/')))
    $loadClock = [Diagnostics.Stopwatch]::StartNew()
    $loaded = Invoke-RestMethod -Uri ($connection.baseUrl + '/api/models/load') -Method Post -WebSession $session `
        -ContentType 'application/json' -Body (@{ model = [IO.Path]::GetFileName($weightsPath); backend = $Backend } | ConvertTo-Json) -TimeoutSec 180
    if (-not $loaded.ok) { throw 'Model load failed.' }
    $loadSeconds = $loadClock.Elapsed.TotalSeconds
    $uiChecks = 0
    $navigation = 'not run (Debug hooks required)'
    $upload = 'not run (Debug hooks required)'
    if ($Configuration -eq 'Debug') {
        # The startup demo sends through the actual composer, navigates to Models,
        # then returns and measures rendered answer text. Wait for it before HTTP work.
        do {
            if ($app.HasExited -or $clock.Elapsed.TotalSeconds -gt 180) { throw 'Native navigation probe did not finish.' }
            Start-Sleep -Milliseconds 500
            $log = Get-Content -LiteralPath (Join-Path $evidenceRoot 'stdout.log') -Raw
        } while ($log -notmatch 'navcheck ok back in the chat' -and $log -notmatch 'navcheck FAIL')
        if ($log -match '(uicheck|uploadcheck|navcheck) FAIL') { throw 'A native WebView probe failed; inspect stdout.log.' }
        $uiChecks = @([regex]::Matches($log, 'uicheck [^\r\n]+ ok')).Count
        if ($uiChecks -lt 12 -or $log -notmatch 'uploadcheck ok') { throw 'Native WebView/upload coverage is incomplete.' }
        # The navigation assertion can finish while the answer is still streaming.
        do {
            if ($clock.Elapsed.TotalSeconds -gt 240) { throw 'The WebView demo did not finish within four minutes.' }
            Start-Sleep -Milliseconds 500
            $queue = Invoke-RestMethod -Uri ($connection.baseUrl + '/api/queue/status') -WebSession $session -TimeoutSec 15
        } while ($queue.busy -or $queue.pending_requests -gt 0)
        $navigation = 'passed'
        $upload = 'passed'
    }
    $benchmarkArgs = @((Join-Path $PSScriptRoot 'tensoragent-desktop-bench.py'), '--connection', $connectionPath,
        '--out', (Join-Path $evidenceRoot 'benchmark'), '--throughput-runs', '3')
    if ($projectorPath) { $benchmarkArgs += '--vision' }
    if ($Tools) { $benchmarkArgs += '--tools' }
    & $Python @benchmarkArgs *> (Join-Path $evidenceRoot 'benchmark.log')
    $benchExit = $LASTEXITCODE
    $app.Refresh()
    @{ startupSeconds = $startupSeconds; modelLoadSeconds = $loadSeconds; uiChecks = $uiChecks;
       navigation = $navigation; upload = $upload; benchmarkExitCode = $benchExit; configuration = $Configuration;
       workingSetBytes = $app.WorkingSet64; peakWorkingSetBytes = $app.PeakWorkingSet64; privateBytes = $app.PrivateMemorySize64;
       model = $weightsPath; projector = $projectorPath; backend = $Backend; context = $Context;
       limits = 'Debug uses synthetic WebView gestures; Release runs HTTP benchmarks only. No manual visual, physical camera or native file-dialog verification.' } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidenceRoot 'summary.json') -Encoding UTF8
    if ($benchExit -ne 0) { throw "HTTP benchmark failed with exit code $benchExit; inspect benchmark/results.json." }
    Get-Content -LiteralPath (Join-Path $evidenceRoot 'summary.json')
} finally {
    # End only this test process. A forced stop also leaves durable transcripts for
    # a later restart test; it is not evidence of graceful application shutdown.
    if ($app -and -not $app.HasExited) { Stop-Process -Id $app.Id }
}
