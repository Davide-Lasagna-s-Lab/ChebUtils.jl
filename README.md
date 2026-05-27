# ChebUtils.jl

**ChebUtils** provides lightweight Chebyshev spectral utilities for Julia:
Chebyshev–Gauss–Lobatto nodes, first- and second-order spectral differentiation
matrices, Clenshaw-Curtis quadrature weights, and their adjoints —
all with allocation-free N-dimensional `mul!` support.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Davide-Lasagna-s-Lab/ChebUtils.jl")
```

## Quick start

```julia
using ChebUtils, LinearAlgebra

N = 64
y  = chebpts(N)          # CGL nodes on [-1, 1], y[1] = +1, y[N] = -1
D  = chebdiff(N)         # N×N first-derivative matrix
DD = chebddiff(N)        # N×N second-derivative matrix
w  = chebws(N)           # Clenshaw-Curtis quadrature weights

# Differentiate a 1-D function: d/dy exp(y) ≈ exp(y)
f  = exp.(y)
df = similar(f)
mul!(df, D, f)
@assert df ≈ f           # spectral accuracy

# Integrate: ∫₋₁¹ exp(y) dy = e - 1/e
@assert abs(dot(w, f) - (exp(1) - exp(-1))) < 1e-14

# Differentiate along dimension 2 of a 3-D array
F  = randn(8, N, 8)
dF = similar(F)
mul!(dF, D, F, Val(2))

# Weighted adjoint for the Clenshaw-Curtis inner product
Dw = adjoint(D, w)       # D† = W⁻¹ D' W
```

## Features

| Function / Type | Description |
|----------------|-------------|
| `chebpts(N)` | N Chebyshev–Gauss–Lobatto nodes on `[-1, 1]` |
| `chebdiff(N)` | First-order spectral differentiation matrix |
| `chebddiff(N)` | Second-order spectral differentiation matrix |
| `chebws(N)` | Clenshaw-Curtis quadrature weights |
| `ChebDiff{T,N}` | Matrix type wrapping the dense differentiation matrix |
| `AdjointChebDiff{T,N}` | Precomputed (weighted) adjoint |
| `mul!(y, D, x, Val(DIM))` | Differentiate N-D array along dimension `DIM` |
| `lu!(D)` | In-place LU for boundary-value problems |

## Grid convention

Nodes are ordered from `+1` (index 1) to `−1` (index `N`), following the
Trefethen convention used in *Spectral Methods in MATLAB* (SIAM, 2000).
The boundary nodes are always at indices `1` and `N`.

## Only dependency

`LinearAlgebra` (Julia standard library).
