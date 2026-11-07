param(
    [ValidateSet('smoke', 'train', 'tune', 'volume')][string]$Mode = 'smoke',
    [string]$Seeds = '42',
    [string]$Models = '',
    [string]$DriverMemory = '2g',
    [string]$SparkHome = $env:SPARK_HOME,
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$HadoopHome = $env:HADOOP_HOME,
    [string[]]$AppArgs = @()
)
# Local-mode launcher using the configured runtime.
$ErrorActionPreference = 'Stop'
if (-not $SparkHome -or -not (Test-Path -LiteralPath (Join-Path $SparkHome 'jars'))) {
    throw 'Set SPARK_HOME or pass -SparkHome with a Spark 3.5.x installation.'
}
& (Join-Path $PSScriptRoot 'run-models.ps1') -Mode $Mode -SparkHome $SparkHome -JavaHome $JavaHome -HadoopHome $HadoopHome -DriverMemory $DriverMemory -Seeds $Seeds -Models $Models -AppArgs $AppArgs
