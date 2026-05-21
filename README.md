# ChebUtils.jl

Lightweight Julia utilities for **Chebyshev spectral methods** on
Chebyshev–Gauss–Lobatto (CGL) grids.

The package provides:

- **CGL node locations** via `chebpts`
- **First- and second-order spectral differentiation matrices** via `chebdiff` / `chebddiff`
- **Clenshaw–Curtis quadrature weights** via `chebws`
- An `AbstractMatrix`-compliant wrapper type `ChebDiff` that integrates with
  `LinearAlgebra` (in-place `mul!` for arbitrary-rank arrays, `lu!`, broadcasting)

---

## Background

On a uniform grid, high-order finite differences suffer from the Runge
phenomenon and condition-number growth.  The Chebyshev–Gauss–Lobatto points

```
xⱼ = cos(π(j−1)/(N−1)),    j = 1, …, N
```

cluster towards ±1 and are the natural nodes for polynomial interpolation on
[-1, 1].  When a function is smooth, the spectral derivative matrix converges
**exponentially** — the error roughly halves with every extra grid point, far
outpacing any finite-difference scheme.

Differentiation by matrix-vector multiplication (`D * f`) is sometimes called
the **pseudospectral** approach: evaluate the function on the grid, multiply by
the dense differentiation matrix, and read off the derivatives at the same nodes.

---

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/ChebUtils.jl")
```

Or from the REPL package manager:

```
pkg> add https://github.com/Davide-Lasagna-s-Lab/ChebUtils.jl
```

---

## Quick start

```julia
using ChebUtils, LinearAlgebra

N  = 32
y  = chebpts(N)          # CGL nodes, length N, ranging from 1 to −1
D  = chebdiff(N)         # N×N first-derivative matrix
DD = chebddiff(N)        # N×N second-derivative matrix
ws = chebws(N)           # Clenshaw–Curtis integration weights

# ── Differentiation ────────────────────────────────────────────────────────
f   = exp.(y)
df  = D  * f             # ≈ exp(y)  (exact up to rounding)
ddf = DD * f             # ≈ exp(y)

# In-place version (no allocation)
dy = similar(f)
mul!(dy, D, f)

# ── Integration ────────────────────────────────────────────────────────────
# ∫₋₁¹ exp(x) dx = e − e⁻¹ ≈ 2.3504
integral = sum(ws .* f)
```

---

## API reference

### `chebpts(N, T=Float64) → Vector{T}`

Returns the N Chebyshev–Gauss–Lobatto node locations on [-1, 1].
The first element is `1.0` and the last is `-1.0`.

```julia
chebpts(5)
# [1.0, 0.707…, 0.0, −0.707…, −1.0]
```

---

### `chebdiff(N, T=Float64) → ChebDiff{T,N}`

First-order spectral differentiation matrix of size N×N.

Off-diagonal entries follow the Trefethen formula:

```
Dᵢⱼ = (cᵢ/cⱼ) · (−1)^(i+j) / (xᵢ − xⱼ)    (i ≠ j)
```

where `cᵢ = 2` at the endpoints and `1` elsewhere.  Diagonal entries are set
by the negative-sum trick so that `D * 1 = 0` exactly.

---

### `chebddiff(N, T=Float64) → ChebDiff{T,N}`

Second-order spectral differentiation matrix of size N×N.

The entries are derived from `D = chebdiff(N)` via the **Fornberg recurrence**,
which is more numerically stable than naively computing `D * D`:

```
DDᵢⱼ = 2·Dᵢⱼ·(Dᵢᵢ − 1/(xᵢ − xⱼ))          (i ≠ j)
DDᵢᵢ = 2·(Dᵢᵢ² + ∑_{k≠i} Dᵢₖ/(xᵢ − xₖ))
```

When you need both `D` and `DD`, build `D` first and pass it to
`chebddiff(D)` to avoid recomputing the grid:

```julia
D  = chebdiff(N)
DD = chebddiff(D)    # reuses D rather than rebuilding it
```

---

### `chebws(N) → Vector{Float64}`

Clenshaw–Curtis quadrature weights for the N-point CGL grid.

```
∫₋₁¹ f(x) dx  ≈  ∑ⱼ wⱼ · f(xⱼ)
```

Exact for all polynomials of degree ≤ N−1; geometrically convergent for
smooth functions.

Algorithm adapted from Trefethen (2000), Program 12.

---

### `ChebDiff{T,N} <: AbstractMatrix{T}`

The matrix type returned by `chebdiff` and `chebddiff`.  It wraps a plain
`Matrix{T}` and implements the full `AbstractMatrix` interface.

Key extra methods:

| Method | Description |
|:-------|:------------|
| `parent(D)` | Return the underlying `Matrix{T}` without copying |
| `lu!(D)` | In-place LU factorisation (forwards to the underlying matrix) |
| `mul!(y, D, x)` | In-place matrix–vector product |
| `mul!(y, D, x, Val(dim))` | Differentiate N-D array `x` along dimension `dim` |

---

## Multi-dimensional differentiation

For problems with multiple spatial dimensions (e.g. a 3-D field on a
z × y × t grid), `mul!` with a `Val` dimension argument differentiates
along any axis without allocating temporaries:

```julia
Ny, Nz, Nt = 32, 64, 16
D  = chebdiff(Ny)

# 3-D field with CGL grid along the second axis (y)
x  = rand(Nz, Ny, Nt)
dy = similar(x)
mul!(dy, D, x, Val(2))    # ∂x/∂y at every (z, t) point
```

The `@generated` implementation unrolls the loop nest at **compile time** for
each concrete dimensionality, so there is no runtime overhead from the
dimension-selection machinery.

---

## Boundary-value problems with `lu!`

Chebyshev differentiation matrices are singular by construction (constant
functions are in the null space of `D`).  To solve a BVP, replace the rows
corresponding to boundary nodes with the constraint equations before
factorising:

```julia
N  = 64
DD = chebddiff(N)
y  = chebpts(N)

# Poisson equation: u''(y) = f(y),  u(±1) = 0
rhs = @. sin(π*y)            # right-hand side

# Impose Dirichlet BCs by overwriting boundary rows
DD[1,   :] .= 0;  DD[1,   1]   = 1;  rhs[1]   = 0
DD[end, :] .= 0;  DD[end, end] = 1;  rhs[end] = 0

u = lu!(DD) \ rhs            # solve in-place
```

---

## Numerical precision

All factory functions accept an optional type argument, making it easy to
switch between precisions:

```julia
D32  = chebdiff(32, Float32)   # single precision
D64  = chebdiff(32)            # double precision (default)
DBig = chebdiff(32, BigFloat)  # arbitrary precision
```

For non-real types (e.g. `ComplexF64`) a plain `Matrix{T}` is returned instead
of a `ChebDiff`.

---

## Running the tests

```
julia --project=test test/runtests.jl
```

The test suite covers:

- Constructor type correctness and promotion
- CGL node values at known angles
- First- and second-order matrix entries against analytic values
- Clenshaw–Curtis integration of `exp` and odd polynomials
- In-place `mul!` for vectors, 3-D arrays (cubes), and 4-D arrays (hypercubes)
- LU factorisation round-trip

---

## References

1. Trefethen, L.N. (2000). *Spectral Methods in MATLAB*. SIAM.
2. Fornberg, B. (1998). *A Practical Guide to Pseudospectral Methods*. Cambridge University Press.
3. Boyd, J.P. (2001). *Chebyshev and Fourier Spectral Methods* (2nd ed.). Dover.

---

## Authors

Thomas Burton `<tburton5572@gmail.com>`

## License

See [Project.toml](Project.toml) for version and dependency information.
