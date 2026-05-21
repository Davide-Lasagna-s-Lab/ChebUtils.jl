"""
    ChebUtils

Utilities for spectral methods on Chebyshev–Gauss–Lobatto (CGL) grids.

The CGL points on [-1, 1] are defined as

    xⱼ = cos(π(j-1)/(N-1)),   j = 1, …, N

with x₁ = 1 and xₙ = -1.  They cluster towards the endpoints, giving
geometric convergence for smooth functions and avoiding the Runge phenomenon
that plagues uniform grids.

# Exported API

| Symbol       | Description                                               |
|:------------ |:--------------------------------------------------------- |
| `chebpts`    | CGL node locations                                        |
| `chebdiff`   | First-order Chebyshev differentiation matrix (returns `ChebDiff`) |
| `chebddiff`  | Second-order Chebyshev differentiation matrix             |
| `chebws`     | Clenshaw–Curtis quadrature weights                        |

# Internal types

`ChebDiff{T,N}` is the concrete matrix type wrapping the dense differentiation
matrix.  It integrates with Julia's `LinearAlgebra` so that `mul!`, `lu!` and
broadcasting all work out of the box.

# References

- Trefethen, L.N. (2000). *Spectral Methods in MATLAB*. SIAM.
- Fornberg, B. (1998). *A Practical Guide to Pseudospectral Methods*. Cambridge.
"""
module ChebUtils

using LinearAlgebra

export chebpts, chebdiff, chebddiff, chebws

include("chebdiff.jl")
include("constructors.jl")
include("clencurt.jl")
include("matmul.jl")
include("linalg.jl")

"""
    chebpts(N, T=Float64) -> Vector{T}

Return the `N` Chebyshev–Gauss–Lobatto (CGL) points on [-1, 1]:

    xⱼ = cos(π(j-1)/(N-1)),   j = 1, …, N

The first element is `1.0` and the last is `-1.0`.  The optional type
parameter `T` controls the floating-point precision of the returned vector.

# Arguments
- `N::Int`: number of grid points (must be ≥ 2).
- `T::Type`: element type of the output (default `Float64`).

# Examples
```julia
julia> chebpts(5)
5-element Vector{Float64}:
  1.0
  0.7071067811865476
  6.123233995736766e-17   # ≈ 0
 -0.7071067811865475
 -1.0

julia> chebpts(5, Float32)
5-element Vector{Float32}:
  1.0
  0.70710677
  0.0
 -0.70710677
 -1.0
```

See also [`chebdiff`](@ref), [`chebws`](@ref).
"""
chebpts(N::Int, ::Type{T}=Float64) where {T} = T.(cos.((0:(N - 1))π/(N - 1)))

end
