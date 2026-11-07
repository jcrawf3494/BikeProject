param(
    [string]$SparkHome = $env:SPARK_HOME,
    [string]$JavaHome = $env:JAVA_HOME
)
$ErrorActionPreference = 'Stop'
if (-not $SparkHome -or -not (Test-Path -LiteralPath (Join-Path $SparkHome 'jars'))) {
    throw 'Pass -SparkHome pointing to a Spark 3.5.x distribution.'
}
$SparkHome = (Resolve-Path -LiteralPath $SparkHome).Path
if ($JavaHome) { $JavaHome = (Resolve-Path -LiteralPath $JavaHome).Path }
$coreJar = Get-ChildItem -LiteralPath (Join-Path $SparkHome 'jars') -Filter 'spark-core_*.jar' | Select-Object -First 1
if ($coreJar.Name -notmatch 'spark-core_2\.12-3\.5\.') { throw 'This package requires Spark 3.5.x built for Scala 2.12.' }
$javaExe = if ($JavaHome) { Join-Path $JavaHome 'bin\java.exe' } else { (Get-Command java -ErrorAction Stop).Source }
$classDir = Join-Path $PSScriptRoot ('target\classes-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $classDir -Force | Out-Null
$classpath = Join-Path $SparkHome 'jars\*'
$responseFile = Join-Path $classDir 'compiler.args'
$sourcePaths = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'src') -Filter '*.scala' -File
$compilerArgs = @('-classpath', ('"' + $classpath.Replace('\', '/') + '"'), '-deprecation', '-feature', '-encoding', 'UTF-8', '-d', ('"' + $classDir.Replace('\', '/') + '"'))
$compilerArgs += @($sourcePaths | ForEach-Object { '"' + $_.FullName.Replace('\', '/') + '"' })
[System.IO.File]::WriteAllLines($responseFile, $compilerArgs, [System.Text.UTF8Encoding]::new($false))
& $javaExe '-Dscala.usejavacp=true' '-cp' $classpath 'scala.tools.nsc.Main' ('@' + $responseFile)
if ($LASTEXITCODE -ne 0) { throw 'Scala compilation failed.' }
$signature = ($sourcePaths | Sort-Object Name | ForEach-Object { $_.Name + ':' + (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }) -join "`n"
$digest = [System.Security.Cryptography.SHA256]::Create()
try { $tag = ([System.BitConverter]::ToString($digest.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($signature)))).Replace('-', '').Substring(0, 12).ToLower() } finally { $digest.Dispose() }
$jarName = 'bikeshare-models-' + $tag + '.jar'
$jarPath = Join-Path $PSScriptRoot ('target\' + $jarName)
if (Test-Path -LiteralPath $jarPath) {
    Set-Content -LiteralPath (Join-Path $PSScriptRoot 'target\current-models-jar.txt') -Value $jarName
    Write-Host "Using compiled $jarPath"
    return
}
Add-Type -AssemblyName System.IO.Compression
$stream = [System.IO.File]::Open($jarPath, [System.IO.FileMode]::Create)
$archive = [System.IO.Compression.ZipArchive]::new($stream, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    Get-ChildItem -LiteralPath $classDir -File -Recurse -Filter '*.class' | ForEach-Object {
        $relative = $_.FullName.Substring($classDir.Length + 1).Replace('\', '/')
        $entry = $archive.CreateEntry($relative)
        $inputStream = [System.IO.File]::OpenRead($_.FullName)
        $outputStream = $entry.Open()
        try { $inputStream.CopyTo($outputStream) } finally { $inputStream.Dispose(); $outputStream.Dispose() }
    }
} finally { $archive.Dispose(); $stream.Dispose() }
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'target\current-models-jar.txt') -Value $jarName
Write-Host "Compiled $jarPath"
