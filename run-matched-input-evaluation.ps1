param(
    [string]$SparkHome = $env:SPARK_HOME,
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$HadoopHome = $env:HADOOP_HOME,
    [string]$Python = 'python',
    [string]$FullDataDir = (Join-Path $PSScriptRoot 'data\paper_flags_full'),
    [string]$SampleDataDir = (Join-Path $PSScriptRoot 'data\pfc'),
    [string]$OutputDir = (Join-Path $PSScriptRoot 'results'),
    [string]$Master = 'local[2]',
    [string]$DriverMemory = '2g'
)
$ErrorActionPreference = 'Stop'
$settingsPath = Join-Path $PSScriptRoot 'validation\P1_P2_P3_with_paper_inputs.json'
$settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
if (($settings.numeric_features.Count + $settings.categorical_features.Count) -ne 39) {
    throw 'The matched-input protocol requires the expanded 39-input settings.'
}
foreach ($dataFolder in @($FullDataDir, $SampleDataDir)) {
    if (-not (Test-Path -LiteralPath $dataFolder -PathType Container)) {
        throw "Prepared private data folder is missing: $dataFolder"
    }
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
$FullDataDir = (Resolve-Path -LiteralPath $FullDataDir).Path
$SampleDataDir = (Resolve-Path -LiteralPath $SampleDataDir).Path
$batchName = 'm39-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$batch = Join-Path $OutputDir $batchName
$primary = Join-Path $batch 'primary'
$samples = Join-Path $batch 'samples'
$xgb = Join-Path $batch 'xgb'
foreach ($folder in @($primary, $samples, $xgb)) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    Copy-Item -LiteralPath $settingsPath -Destination (Join-Path $folder 'frozen_selected_settings.json')
}
function Write-State([string]$Path, $Value) {
    $json = $Value | ConvertTo-Json -Depth 20
    [System.IO.File]::WriteAllText($Path, $json + "`n", [System.Text.UTF8Encoding]::new($false))
}
Write-State (Join-Path $batch 'protocol.json') @{
    started_utc = (Get-Date).ToUniversalTime().ToString('o')
    feature_count = 39; test_rows = 133553; seeds = @(42,43,44)
    expected_spark_cases = 27; expected_xgboost_cases = 9
    training_rows = @{ full = 536406; similarity = 10354; stratified = 10354 }
    settings_sha256 = (Get-FileHash -LiteralPath $settingsPath -Algorithm SHA256).Hash.ToLowerInvariant()
    scope = 'New local rerun; historical published attestations are preserved'
}
foreach ($cohort in @('full', 'similarity', 'stratified')) {
    foreach ($seed in @(42,43,44)) {
        if ($cohort -eq 'full') {
            $data = $FullDataDir
            $stateFolder = $primary
        } else {
            $data = Join-Path $SampleDataDir (Join-Path $cohort ('seed-' + $seed))
            $stateFolder = $samples
        }
        foreach ($model in @('P3', 'P1', 'P2')) {
            $key = $model + '_seed_' + $seed
            if ($cohort -ne 'full') { $key = $cohort + '_' + $key }
            $statePath = Join-Path $stateFolder ($key + '.json')
            $out = Join-Path $batch ($cohort.Substring(0,2) + '-' + $model + '-' + $seed)
            $state = @{ status = 'running'; cohort = $cohort; model = $model; seed = $seed }
            Write-State $statePath $state
            try {
                & (Join-Path $PSScriptRoot 'run-models.ps1') -Mode train -Models $model -Seeds ([string]$seed) `
                    -SparkHome $SparkHome -JavaHome $JavaHome -HadoopHome $HadoopHome `
                    -DataDir $data -OutputDir $out -Master $Master -DriverMemory $DriverMemory `
                    -AppArgs @('--evaluation','test','--fraction','1.0','--settings-file',$settingsPath,
                               '--tree-encoding','native','--feature-variant','standard','--search-budget','extended',
                               '--numeric-features',($settings.numeric_features -join ','),
                               '--categorical-features',($settings.categorical_features -join ','))
                $completed = @(Get-ChildItem -LiteralPath $out -Directory | Where-Object {
                    Test-Path -LiteralPath (Join-Path $_.FullName 'results.json')
                })
                if ($completed.Count -ne 1) { throw 'Expected exactly one completed run in the fresh output directory.' }
                $results = @(Get-Content -LiteralPath (Join-Path $completed[0].FullName 'results.json') -Raw | ConvertFrom-Json)
                $expectedRows = if ($cohort -eq 'full') { 536406 } else { 10354 }
                if ($results.Count -ne 1 -or $results[0].training_rows -ne $expectedRows -or $results[0].evaluation_rows -ne 133553) {
                    throw 'Completed result did not match the declared training/test cohorts.'
                }
                $state.status = 'completed'
                $state.run = $completed[0].FullName
                Write-State $statePath $state
            } catch {
                $state.status = 'failed'
                Write-State $statePath $state
                throw
            }
        }
    }
    if ($cohort -eq 'full') {
        Write-State (Join-Path $primary 'batch_completed.json') @{ model_records = 9 }
    }
}
Write-State (Join-Path $samples 'batch_completed.json') @{ model_records = 18 }
& $Python (Join-Path $PSScriptRoot 'tools\run_matched_feature_xgboost.py') `
    --package $PSScriptRoot --metadata-dir $xgb --primary-state $primary --sample-state $samples `
    --full-data-dir $FullDataDir --sample-data-dir $SampleDataDir
if ($LASTEXITCODE -ne 0) { throw 'Matched-input XGBoost control failed; inspect its saved state.' }
Write-State (Join-Path $batch 'batch_completed.json') @{ spark_model_records = 27; xgboost_model_records = 9 }
Write-Host "Completed matched-input evaluation. Local state: $batch"
