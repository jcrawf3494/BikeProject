param(
    [ValidateSet('smoke', 'train', 'tune', 'volume')][string]$Mode = 'smoke',
    [string]$SparkHome = $env:SPARK_HOME,
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$HadoopHome = $env:HADOOP_HOME,
    [string]$Master = 'local[2]',
    [string]$DriverMemory = '2g',
    [ValidatePattern('^[1-9][0-9]*[kKmM]$')][string]$JavaStackSize = '4m',
    [string]$Models = '',
    [string]$Seeds = '42',
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
if (-not $DataDir) { $DataDir = Join-Path $PSScriptRoot 'data' }
if (-not $OutputDir) { $OutputDir = Join-Path $PSScriptRoot 'results' }
$env:SPARK_HOME = $SparkHome
$env:JAVA_HOME = $JavaHome
if ($HadoopHome) {
    $env:HADOOP_HOME = $HadoopHome
    $env:PATH = (Join-Path $HadoopHome 'bin') + ';' + $env:PATH
}
if (-not $env:SPARK_LOCAL_IP -and $Master -like 'local*') { $env:SPARK_LOCAL_IP = '127.0.0.1' }
$jarPath = Join-Path $PSScriptRoot 'target\bikeshare-models.jar'
$pointer = Join-Path $PSScriptRoot 'target\current-models-jar.txt'
if (Test-Path -LiteralPath $pointer) {
    $jarName = (Get-Content -LiteralPath $pointer -Raw).Trim()
    if ($jarName -notmatch '^bikeshare-models-[a-f0-9]+\.jar$') { throw 'Invalid compiled JAR pointer.' }
    $jarPath = Join-Path $PSScriptRoot ('target\' + $jarName)
}
if (-not (Test-Path -LiteralPath $jarPath)) { & (Join-Path $PSScriptRoot 'compile-models.ps1') -SparkHome $SparkHome -JavaHome $JavaHome }
if (Test-Path -LiteralPath $pointer) { $jarPath = Join-Path $PSScriptRoot ('target\' + (Get-Content -LiteralPath $pointer -Raw).Trim()) }
$arguments = @('--master', $Master, '--driver-memory', $DriverMemory, '--conf', 'spark.ui.enabled=false', '--class', 'TrainBikeShareModels', $jarPath,
    '--data-dir', $DataDir, '--output-dir', $OutputDir, '--mode', $Mode, '--seeds', $Seeds)
if ($Models) { $arguments += @('--models', $Models) }
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
