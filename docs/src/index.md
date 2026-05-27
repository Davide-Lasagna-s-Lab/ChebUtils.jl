# ChebUtils.jl

**ChebUtils** provides lightweight Chebyshev spectral utilities for Julia:
Chebyshev–Gauss–Lobatto nodes, first- and second-order spectral differentiation
matrices, Clenshaw-Curtis quadrature weights, and their adjoints —
all with allocation-free N-dimensional `mul!` support.

Only dependency: `LinearAlgebra` (Julia standard library).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Davide-Lasagna-s-Lab/ChebUtils.jl")
```

## Quick start

```julia
using ChebUtils, LinearAlgebra

N = 64
y  = chebpts(N)          # CGL nodes on [-1, 1]: y[1]=+1, y[N]=-1
D  = chebdiff(N)         # N×N first-derivative matrix
DD = chebddiff(N)        # N×N second-derivative matrix
w  = chebws(N)           # Clenshaw-Curtis weights

# Differentiate: d/dy exp(y) = exp(y)
f  = exp.(y)
df = similar(f)
mul!(df, D, f)

# Integrate: ∫₋₁¹ exp(y) dy = e - 1/e
dot(w, f) ≈ exp(1) - exp(-1)   # true

# Differentiate a 3-D field along its second dimension
F  = randn(8, N, 8)
mul!(similar(F), D, F, Val(2))

# Weighted adjoint for the Clenshaw-Curtis inner product
Dw = adjoint(D, w)       # D† = W⁻¹ D' W
```

## Contents

```@contents
Pages = ["guide.md", "api.md"]
Depth = 2
```
