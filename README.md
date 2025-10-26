# Equivariant Structured Positional Rotations

This workspace contains a LaTeX manuscript that formalizes the BEG-0 through BEG-6 development for structured positional rotations in attention heads.

## Prerequisites

Install a LaTeX distribution that bundles `pdflatex`. On Windows one of the following is recommended:

- [MiKTeX](https://miktex.org/download)
- [TeX Live](https://www.tug.org/texlive/acquire-netinstall.html)

After installation, ensure the LaTeX binaries are on your `PATH`. You can verify by running the command below in **Windows PowerShell**:

```powershell
pdflatex --version
```

## Building the PDF

A helper PowerShell script is provided that compiles `main.tex` twice (to resolve references) and moves the output into the `build` directory.

```powershell
./build.ps1
```

The generated PDF will be located at `build/main.pdf`.

## Manual Compilation

If you prefer to run `pdflatex` manually, execute the following commands from the project root:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Cleaning auxiliary files:

```powershell
Remove-Item *.aux, *.log, *.out, *.toc -ErrorAction Ignore
```

## VS Code Extensions

The workspace recommends installing the official LaTeX Workshop extension for an improved editing experience.
