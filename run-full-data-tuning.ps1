param(
    [string]$Models = 'P1',
    [long]$Seed = 42,
    [ValidateSet('standard', 'extended')][string]$SearchBudget = 'extended',
    [ValidateSet('onehot', 'native')][string]$TreeEncoding = 'native',
    [ValidateSet('standard', 'interactions')][string]$FeatureVariant = 'standard',
    [string]$DriverMemory = '2g',
    [string]$SparkHome = $env:SPARK_HOME,
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$HadoopHome = $env:HADOOP_HOME,
    [switch]$Help
)
$ErrorActionPreference = 'Stop'
if ($Help) {
    Write-Output 'Run full-training-pool, training-validation tuning. Default: P1, extended grid, native tree categories, standard features.'
    Write-Output "Example: .\run-full-data-tuning.ps1 -Models 'P1,P2,P3' -FeatureVariant interactions"
    Write-Output 'Models run sequentially, with separate result folders and settings files. The original project test set is not scored.'
    return
}
$modelList = @($Models.Split(',') | ForEach-Object { $_.Trim().ToUpperInvariant() } | Where-Object { $_ })
if (-not $modelList.Count -or @($modelList | Select-Object -Unique).Count -ne $modelList.Count -or
    @($modelList | Where-Object { $_ -notin @('P1', 'P2', 'P3') }).Count) { throw 'Use P1, P2, P3, or a comma-separated list.' }
foreach ($modelName in $modelList) {
    Write-Host "Full-data tuning: $modelName; seed=$Seed; grid=$SearchBudget; tree encoding=$TreeEncoding; features=$FeatureVariant"
    $modelArguments = @('--fraction', '1.0', '--evaluation', 'validation', '--search-budget', $SearchBudget,
        '--tree-encoding', $TreeEncoding, '--feature-variant', $FeatureVariant)
    if ($SparkHome) {
        & (Join-Path $PSScriptRoot 'run-models.ps1') -Mode tune -Models $modelName -Seeds ([string]$Seed) `
            -SparkHome $SparkHome -JavaHome $JavaHome -HadoopHome $HadoopHome -DriverMemory $DriverMemory -AppArgs $modelArguments
    } else {
        & (Join-Path $PSScriptRoot 'run-local-models.ps1') -Mode tune -Models $modelName -Seeds ([string]$Seed) `
            -DriverMemory $DriverMemory -AppArgs $modelArguments
    }
}
