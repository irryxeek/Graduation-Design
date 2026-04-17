param(
    [string]$OutputPath = "docs/thesis/figures/data_processing_pipeline_scientific_schematics.png",
    [ValidateSet("journal", "conference", "poster", "presentation", "report", "grant", "thesis", "preprint", "default")]
    [string]$DocType = "thesis",
    [ValidateRange(1, 2)]
    [int]$Iterations = 2
)

$ErrorActionPreference = "Stop"

$skillRoot = "C:\Users\sense\.agents\skills\scientific-schematics"
$generator = Join-Path $skillRoot "scripts\generate_schematic.py"
$promptFile = "docs/thesis/figures/data_processing_pipeline_scientific_schematics_prompt.md"

if (-not (Test-Path $generator)) {
    throw "scientific-schematics generator not found: $generator"
}

if (-not (Test-Path $promptFile)) {
    throw "Prompt file not found: $promptFile"
}

if (-not $env:OPENROUTER_API_KEY) {
    throw "OPENROUTER_API_KEY is not set. Set the API key first, then rerun this script."
}

$env:PYTHONIOENCODING = "utf-8"

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir) {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

$prompt = Get-Content -Raw -Path $promptFile
$prompt = $prompt -replace "`r?`n", " "
$prompt = $prompt -replace "\s{2,}", " "
$prompt = $prompt.Replace('"', "'")

Write-Host "Generating thesis figure with scientific-schematics..."
Write-Host "Output: $OutputPath"
Write-Host "DocType: $DocType"
Write-Host "Iterations: $Iterations"

& py -3 $generator $prompt -o $OutputPath --doc-type $DocType --iterations $Iterations -v
