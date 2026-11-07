param(
    [string]$SparkHome = $env:SPARK_HOME,
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$HadoopHome = $env:HADOOP_HOME,
    [string]$Master = 'local[2]',
    [string]$DriverMemory = '2g',
    [ValidatePattern('^[1-9][0-9]*[kKmM]$')][string]$JavaStackSize = '4m',
    [string]$InitFile = ''
)
$ErrorActionPreference = 'Stop'
if (-not $SparkHome -or -not (Test-Path -LiteralPath (Join-Path $SparkHome 'jars'))) {
    throw 'Pass -SparkHome with the path to a Spark 3.5.x distribution.'
}
$coreJar = Get-ChildItem -LiteralPath (Join-Path $SparkHome 'jars') -Filter 'spark-core_*.jar' | Select-Object -First 1
if ($coreJar.Name -notmatch 'spark-core_2\.12-3\.5\.') { throw 'Use Spark 3.5.x with Scala 2.12.' }
if (-not $JavaHome) { throw 'Pass -JavaHome with a supported JDK path.' }
if (-not $HadoopHome -or -not (Test-Path -LiteralPath (Join-Path $HadoopHome 'bin\hadoop.dll'))) {
    throw 'Pass -HadoopHome with compatible Windows Hadoop helpers under its bin folder.'
}
$SparkHome = (Resolve-Path -LiteralPath $SparkHome).Path
$JavaHome = (Resolve-Path -LiteralPath $JavaHome).Path
$HadoopHome = (Resolve-Path -LiteralPath $HadoopHome).Path
$env:SPARK_HOME = $SparkHome
$env:JAVA_HOME = $JavaHome
$env:HADOOP_HOME = $HadoopHome
$env:PATH = (Join-Path $HadoopHome 'bin') + ';' + (Join-Path $JavaHome 'bin') + ';' + $env:PATH
$env:BIKESHARE_PROJECT = $PSScriptRoot.Replace('\', '/')
if (-not $env:SPARK_LOCAL_IP -and $Master -like 'local*') { $env:SPARK_LOCAL_IP = '127.0.0.1' }
$jarPath = Join-Path $PSScriptRoot 'target\bikeshare-models.jar'
$pointer = Join-Path $PSScriptRoot 'target\current-models-jar.txt'
if (Test-Path -LiteralPath $pointer) {
    $jarName = (Get-Content -LiteralPath $pointer -Raw).Trim()
    if ($jarName -notmatch '^bikeshare-models-[a-f0-9]+\.jar$') { throw 'Invalid compiled JAR pointer.' }
    $jarPath = Join-Path $PSScriptRoot ('target\' + $jarName)
}
if (-not (Test-Path -LiteralPath $jarPath)) {
    & (Join-Path $PSScriptRoot 'compile-models.ps1') -SparkHome $SparkHome -JavaHome $JavaHome
}
if (Test-Path -LiteralPath $pointer) { $jarPath = Join-Path $PSScriptRoot ('target\' + (Get-Content -LiteralPath $pointer -Raw).Trim()) }
$nativeBin = (Join-Path $HadoopHome 'bin').Replace('\', '/')
$arguments = @('--master', $Master, '--driver-memory', $DriverMemory, '--conf', 'spark.ui.enabled=false',
    '--driver-java-options', ('-Xss' + $JavaStackSize + ' -Djava.library.path="' + $nativeBin + '"'),
    '--conf', ('spark.executor.extraJavaOptions=-Xss' + $JavaStackSize), '--jars', $jarPath)
if ($InitFile) { $arguments += @('-i', (Resolve-Path -LiteralPath $InitFile).Path) }
Write-Host 'The model JAR is loaded automatically. At scala>, use sys.env("BIKESHARE_PROJECT") for the project folder.'
& (Join-Path $SparkHome 'bin\spark-shell.cmd') @arguments
if ($LASTEXITCODE -ne 0) { throw "Spark Shell exited with code $LASTEXITCODE." }
