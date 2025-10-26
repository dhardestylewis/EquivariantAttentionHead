param(
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

if ($Clean) {
    Remove-Item -Path *.aux, *.log, *.out, *.toc -ErrorAction SilentlyContinue
    if (Test-Path build) {
        Remove-Item -Path build -Recurse -Force
    }
    Write-Host "Clean complete." -ForegroundColor Green
    exit 0
}

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    Write-Error "pdflatex is not available on PATH. Install MiKTeX or TeX Live and ensure pdflatex is accessible."
}

$buildDir = Join-Path (Get-Location) 'build'
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

$latexArgs = @('-interaction=nonstopmode', '-halt-on-error', 'main.tex')

Write-Host "Running pdflatex (pass 1)..."
& pdflatex @latexArgs | Out-Host

Write-Host "Running pdflatex (pass 2)..."
& pdflatex @latexArgs | Out-Host

$generatedPdf = Join-Path (Get-Location) 'main.pdf'
if (Test-Path $generatedPdf) {
    Move-Item -Path $generatedPdf -Destination (Join-Path $buildDir 'main.pdf') -Force
    Write-Host "Build complete: $buildDir\main.pdf" -ForegroundColor Green
} else {
    Write-Warning "Compilation finished but main.pdf was not found. Check pdflatex output for errors."
}
