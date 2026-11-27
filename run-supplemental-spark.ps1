param(
    [ValidateSet('smoke', 'train', 'tune')][string]$Mode = 'train',
    [string]$SparkHome = $env:SPARK_HOME,
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$HadoopHome = $env:HADOOP_HOME,
    [string]$Master = 'local[2]',
    [string]$DriverMemory = '2g',
    [ValidatePattern('^[1-9][0-9]*[kKmM]$')][string]$JavaStackSize = '4m',
    [string]$Models = 'P4,P5',
    [string]$Seeds = '42,43,44',
    [string]$DataDir = '',
    [string]$OutputDir = '',
    [string[]]$AppArgs = @()
)
$ErrorActionPreference = 'Stop'
if (-not $SparkHome) { throw 'Pass -SparkHome pointing to Spark 3.5.x.' }
if (-not $JavaHome) { throw 'Pass -JavaHome pointing to a supported JDK (11 or 17 recommended).' }
$SparkHome = (Resolve-Path -LiteralPath $SparkHome).Path
$JavaHome = (Resolve-Path -LiteralPath $JavaHome).Path
if ($HadoopHome) { $HadoopHome = (Resolve-Path -LiteralPath $HadoopHome).Path }
if (-not $DataDir) { $DataDir = Join-Path $PSScriptRoot 'data\paper_flags_full' }
if (-not $OutputDir) { $OutputDir = Join-Path $PSScriptRoot 'results' }
$env:SPARK_HOME = $SparkHome
$env:JAVA_HOME = $JavaHome
if ($HadoopHome) {
    $env:HADOOP_HOME = $HadoopHome
    $env:PATH = (Join-Path $HadoopHome 'bin') + ';' + $env:PATH
}
if (-not $env:SPARK_LOCAL_IP -and $Master -like 'local*') { $env:SPARK_LOCAL_IP = '127.0.0.1' }
$jarPath = Join-Path $PSScriptRoot 'target\bikeshare-supplemental.jar'
$pointer = Join-Path $PSScriptRoot 'target\current-supplemental-jar.txt'
if (Test-Path -LiteralPath $pointer) {
    $jarName = (Get-Content -LiteralPath $pointer -Raw).Trim()
    if ($jarName -notmatch '^bikeshare-supplemental-[a-f0-9]+\.jar$') { throw 'Invalid compiled JAR pointer.' }
    $jarPath = Join-Path $PSScriptRoot ('target\' + $jarName)
}
if (-not (Test-Path -LiteralPath $jarPath)) { & (Join-Path $PSScriptRoot 'compile-supplemental-models.ps1') -SparkHome $SparkHome -JavaHome $JavaHome }
if (Test-Path -LiteralPath $pointer) { $jarPath = Join-Path $PSScriptRoot ('target\' + (Get-Content -LiteralPath $pointer -Raw).Trim()) }
$arguments = @('--master', $Master, '--driver-memory', $DriverMemory, '--conf', 'spark.ui.enabled=false', '--class', 'TrainSupplementalSparkModels', $jarPath,
    '--data-dir', $DataDir, '--output-dir', $OutputDir, '--mode', $Mode, '--seeds', $Seeds)
if ($Models) { $arguments += @('--models', $Models) }
$evaluation = if ($Mode -eq 'train') { 'test' } else { 'validation' }
$arguments += @('--evaluation', $evaluation, '--fraction', '1.0', '--tree-encoding', 'onehot')
$arguments += $AppArgs
$driverOptions = '-Xss' + $JavaStackSize
# Select the explicit project's Hadoop helpers instead of an older DLL elsewhere on PATH.
if ($HadoopHome -and $Master -like 'local*') {
    $nativeBin = (Join-Path $HadoopHome 'bin').Replace('\', '/')
    $driverOptions += ' -Djava.library.path="' + $nativeBin + '"'
}
$arguments = @('--driver-java-options', $driverOptions, '--conf', ('spark.executor.extraJavaOptions=-Xss' + $JavaStackSize)) + $arguments
& (Join-Path $SparkHome 'bin\spark-submit.cmd') @arguments
if ($LASTEXITCODE -ne 0) { throw "Spark exited with code $LASTEXITCODE. See the error above." }
