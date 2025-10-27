# Equivariant Structured Positional Rotations

This workspace contains LaTeX manuscripts that formalize the relative-position attention theorem via commuting skew-symmetric generators, proven through **two independent approaches**:

## Proof Approaches

### 1. **proof-algebraic-detailed.tex** (2040 lines)
- **Style**: Detailed step-by-step derivation with full algebraic expansions
- **Structure**: Sections BEG-0 through BEG-6 with narrative flow
- **Target**: Complete textbook-style exposition showing every matrix manipulation
- **Use case**: Understanding the detailed mechanics of each transformation

### 2. **proof-axiomatic-compact.tex** (692 lines)
- **Style**: Compact axiomatic proof with boxed statements and dependency tracking
- **Structure**: Numbered statements S0.1–S8.2 with explicit `[deps: ...]` annotations
- **Target**: Formal verification-ready presentation with minimal redundancy
- **Key features**:
  - Simultaneous block diagonalization theorem (S2.2) with proof sketch
  - Complete S0 foundation lemmas (determinant, exponential, spectral theory)
  - Cayley expressivity fix with eigenvalue -1 restriction
  - Quantitative robustness bounds with error analysis
- **Use case**: Auditing logical dependencies and formal rigor

Both proofs establish theorems **S8.1** (exact relative-position property) and **S8.2** (robust approximation) from first principles.

## Prerequisites

Install a LaTeX distribution that bundles `pdflatex`. On Windows one of the following is recommended:

- [MiKTeX](https://miktex.org/download)
- [TeX Live](https://www.tug.org/texlive/acquire-netinstall.html)

After installation, ensure the LaTeX binaries are on your `PATH`. You can verify by running the command below in **Windows PowerShell**:

```powershell
pdflatex --version
```

## Building the PDFs

Compile either proof document using `pdflatex`:

### Detailed Algebraic Proof
```bash
pdflatex -interaction=nonstopmode -halt-on-error proof-algebraic-detailed.tex
pdflatex -interaction=nonstopmode -halt-on-error proof-algebraic-detailed.tex
```

### Compact Axiomatic Proof
```bash
pdflatex -interaction=nonstopmode -halt-on-error proof-axiomatic-compact.tex
pdflatex -interaction=nonstopmode -halt-on-error proof-axiomatic-compact.tex
```

Run twice to resolve references.

Cleaning auxiliary files:

```powershell
Remove-Item *.aux, *.log, *.out, *.toc -ErrorAction Ignore
```

## VS Code Extensions

The workspace recommends installing the official LaTeX Workshop extension for an improved editing experience.
