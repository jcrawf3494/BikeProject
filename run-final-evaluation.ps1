param(
    [string]$SparkHome = $env:SPARK_HOME,
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$HadoopHome = $env:HADOOP_HOME,
    [string]$DataDir = (Join-Path $PSScriptRoot 'data\paper_flags_full'),
    [string]$OutputDir = (Join-Path $PSScriptRoot 'results'),
    [string]$Master = 'local[2]',
    [string]$DriverMemory = '2g',
    [string]$Seeds = '42,43,44'
)
$ErrorActionPreference = 'Stop'
$settings = Join-Path $PSScriptRoot 'validation\P1_P2_P3_with_paper_inputs.json'
if (-not (Test-Path -LiteralPath $settings)) { throw 'Frozen selected settings are missing.' }
$paperInputs = Get-Content -LiteralPath $settings -Raw | ConvertFrom-Json
$numericInputs = $paperInputs.numeric_features -join ','
$categoricalInputs = $paperInputs.categorical_features -join ','
# Run sequentially to keep peak memory bounded. Imported settings are fixed;
# each run calibrates its cutoff using training rows and preserves the test IDs.
foreach ($model in @('P3', 'P1', 'P2')) {
    $runMode = 'train'
    $modelOutput = Join-Path $OutputDir ('final-' + $model)
    & (Join-Path $PSScriptRoot 'run-models.ps1') `
        -Mode $runMode -Models $model -Seeds $Seeds `
        -SparkHome $SparkHome -JavaHome $JavaHome -HadoopHome $HadoopHome `
        -DataDir $DataDir -OutputDir $modelOutput -Master $Master -DriverMemory $DriverMemory `
        -AppArgs @('--evaluation', 'test', '--fraction', '1.0', '--settings-file', $settings,
                   '--tree-encoding', 'native', '--feature-variant', 'standard',
                   '--search-budget', 'extended', '--numeric-features', $numericInputs,
                   '--categorical-features', $categoricalInputs)
}
