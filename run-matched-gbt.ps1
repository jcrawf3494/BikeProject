param(
    [Parameter(Mandatory=$true)][string]$ConfigFile,
    [string]$SparkHome = $env:SPARK_HOME,
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$HadoopHome = $env:HADOOP_HOME,
    [string]$Master = 'local[4]',
    [string]$DriverMemory = '4g'
)
$ErrorActionPreference = 'Stop'
$SparkHome = (Resolve-Path -LiteralPath $SparkHome).Path
$JavaHome = (Resolve-Path -LiteralPath $JavaHome).Path
$HadoopHome = (Resolve-Path -LiteralPath $HadoopHome).Path
$env:SPARK_HOME = $SparkHome
$env:JAVA_HOME = $JavaHome
$env:HADOOP_HOME = $HadoopHome
$env:SPARK_LOCAL_IP = '127.0.0.1'
$env:PATH = (Join-Path $HadoopHome 'bin') + ';' + (Join-Path $JavaHome 'bin') + ';' + $env:PATH
$pointer = Join-Path $PSScriptRoot 'target\current-matched-jar.txt'
if (-not (Test-Path -LiteralPath $pointer)) {
    & (Join-Path $PSScriptRoot 'compile-matched-models.ps1') -SparkHome $SparkHome -JavaHome $JavaHome
}
$jarName = (Get-Content -LiteralPath $pointer -Raw).Trim()
if ($jarName -notmatch '^bikeshare-matched-[a-f0-9]+\.jar$') { throw 'Invalid matched trainer JAR pointer.' }
$jar = Join-Path $PSScriptRoot ('target\' + $jarName)
$native = (Join-Path $HadoopHome 'bin').Replace('\','/')
$arguments = @('--master',$Master,'--driver-memory',$DriverMemory,'--conf','spark.ui.enabled=false',
    '--driver-java-options',('-Xss4m -Djava.library.path="' + $native + '"'),
    '--conf','spark.executor.extraJavaOptions=-Xss4m','--class','TrainMatchedGBT',$jar,
    '--config',(Resolve-Path -LiteralPath $ConfigFile).Path)
& (Join-Path $SparkHome 'bin\spark-submit.cmd') @arguments
if ($LASTEXITCODE -ne 0) { throw "Matched GBT fitting exited with code $LASTEXITCODE." }
