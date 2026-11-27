param(
    [ValidateSet('smoke','train','tune')][string]$Mode = 'train',
    [string]$Models = 'P4,P5,P6,P7',
    [string]$Seeds = '42,43,44',
    [string]$SparkHome = $env:SPARK_HOME,
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$HadoopHome = $env:HADOOP_HOME,
    [string]$Python = (Join-Path $PSScriptRoot '.venv-additional\Scripts\python.exe'),
    [string]$DataDir = (Join-Path $PSScriptRoot 'data\paper_flags_full'),
    [string]$RolesDir = (Join-Path $PSScriptRoot 'data\supplemental_roles'),
    [string]$OutputDir = (Join-Path $PSScriptRoot 'results'),
    [string]$SparkSettingsFile = '',
    [string]$PythonSettingsFile = ''
)
$ErrorActionPreference = 'Stop'
$requested = @($Models.Split(',') | ForEach-Object { $_.Trim().ToUpperInvariant() })
if ($requested.Count -ne @($requested | Select-Object -Unique).Count -or @($requested | Where-Object { $_ -notin @('P4','P5','P6','P7') }).Count) {
    throw 'Models must be a unique comma-separated selection of P4,P5,P6,P7.'
}
if ($Mode -eq 'tune' -and $Seeds.Split(',').Count -ne 1) {
    throw 'Tune with one declared seed, such as -Seeds 42; then freeze settings for three-seed training.'
}
$sparkModels = @($requested | Where-Object { $_ -in @('P4','P5') })
$pythonModels = @($requested | Where-Object { $_ -in @('P6','P7') })
if ($sparkModels.Count) {
    $app = @()
    if ($SparkSettingsFile) { $app += @('--settings-file', $SparkSettingsFile) }
    & (Join-Path $PSScriptRoot 'run-supplemental-spark.ps1') -Mode $Mode -Models ($sparkModels -join ',') -Seeds $Seeds `
        -SparkHome $SparkHome -JavaHome $JavaHome -HadoopHome $HadoopHome `
        -DataDir $DataDir -OutputDir (Join-Path $OutputDir 'additional-spark') -AppArgs $app
}
if ($pythonModels.Count) {
    $arguments = @((Join-Path $PSScriptRoot 'supplemental\run_python_models.py'), '--mode', $Mode,
                   '--models', ($pythonModels -join ','), '--seeds', $Seeds, '--data-dir', $DataDir,
                   '--roles-dir', $RolesDir, '--output-dir', (Join-Path $OutputDir 'additional-python'))
    if ($PythonSettingsFile) { $arguments += @('--settings-file', $PythonSettingsFile) }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'An additional Python model failed; inspect the saved state and error.' }
}
