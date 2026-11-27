param(
    [ValidateSet('all','tune','evaluate')][string]$Stage = 'all',
    [string]$Python = (Join-Path $PSScriptRoot '.venv-additional\Scripts\python.exe'),
    [string]$SparkHome = $env:SPARK_HOME,
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$HadoopHome = $env:HADOOP_HOME,
    [string]$DataDir = (Join-Path $PSScriptRoot 'data\paper_flags_full'),
    [string]$RolesFile = (Join-Path $PSScriptRoot 'data\supplemental_roles\seed-42.parquet'),
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$TempDir = (Join-Path $PSScriptRoot 'results\matched-temp'),
    [ValidateRange(1,8)][int]$Threads = 4
)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'compile-matched-models.ps1') -SparkHome $SparkHome -JavaHome $JavaHome
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
$jarName = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'target\current-matched-jar.txt') -Raw).Trim()
$artifact = @{
    jar_name = $jarName
    jar_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot ('target\' + $jarName)) -Algorithm SHA256).Hash.ToLowerInvariant()
    scala_source_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'supplemental\matched\TrainMatchedGBT.scala') -Algorithm SHA256).Hash.ToLowerInvariant()
}
$artifactPath = Join-Path $OutputDir 'COMPILED_ARTIFACT.json'
if (Test-Path -LiteralPath $artifactPath) {
    $prior = Get-Content -LiteralPath $artifactPath -Raw | ConvertFrom-Json
    if ($prior.jar_sha256 -ne $artifact.jar_sha256 -or $prior.scala_source_sha256 -ne $artifact.scala_source_sha256) {
        throw 'The compiled worker changed. Preserve this run and choose a new output directory.'
    }
} else {
    [System.IO.File]::WriteAllText($artifactPath,($artifact | ConvertTo-Json) + "`n",[System.Text.UTF8Encoding]::new($false))
}
& $Python (Join-Path $PSScriptRoot 'supplemental\matched\train_matched_models.py') --stage $Stage `
    --data-dir $DataDir --roles-file $RolesFile --output-dir $OutputDir --spark-home $SparkHome `
    --java-home $JavaHome --hadoop-home $HadoopHome --temp-dir $TempDir --threads $Threads
if ($LASTEXITCODE -ne 0) { throw 'Matched tuning stopped; completed cases are preserved and resumable.' }
if ($Stage -ne 'tune') {
    & $Python (Join-Path $PSScriptRoot 'supplemental\matched\audit_study.py') --run $OutputDir --data-dir $DataDir
    if ($LASTEXITCODE -ne 0) { throw 'Training completed, but independent verification needs attention.' }
}
