"""
    ChebUtils

Utilities for spectral methods on Chebyshev–Gauss–Lobatto (CGL) grids.

The CGL points on `[-1, 1]` are defined as

    xⱼ = cos(π(j−1)/(N−1)),   j = 1, …, N

with `x₁ = 1` and `xₙ = −1`.  They cluster towards the endpoints, giving
geometric (exponential) convergence for smooth functions and avoiding the Runge
phenomenon that plagues uniform grids.

# Exported API

| Symbol            | Description                                              |
|:----------------- |:-------------------------------------------------------- |
| `ChebDiff`        | N×N first-order spectral differentiation matrix type     |
| `AdjointChebDiff` | Precomputed adjoint (or weighted adjoint) of `ChebDiff`  |
| `chebpts`         | CGL node locations                                       |
| `chebdiff`        | Construct a first-order `ChebDiff`                       |
| `chebddiff`       | Construct a second-order `ChebDiff`                      |
| `chebws`          | Clenshaw–Curtis quadrature weights                       |

# References

- Trefethen, L.N. (2000). *Spectral Methods in MATLAB*. SIAM.
- Fornberg, B. (1998). *A Practical Guide to Pseudospectral Methods*. Cambridge.
"""
module ChebUtils

using LinearAlgebra

export ChebDiff, AdjointChebDiff
export chebpts, chebdiff, chebddiff, chebws

include("chebdiff.jl")
include("adjoint.jl")
include("constructors.jl")
include("clencurt.jl")
include("matmul.jl")
include("linalg.jl")

"""
    chebpts(N::Int, T::Type=Float64) -> Vector{T}

Return the `N` Chebyshev–Gauss–Lobatto (CGL) nodes on `[-1, 1]`:

```
xⱼ = cos(π(j−1)/(N−1)),   j = 1, …, N
```

The first node is `x₁ = +1` and the last is `xₙ = −1`.  This ordering
matches the row/column layout of [`chebdiff`](@ref) and [`chebddiff`](@ref).

# Examples
```julia
julia> chebpts(3)
3-element Vector{Float64}:
  1.0
  0.0
 -1.0

julia> chebpts(4, Float32)
4-element Vector{Float32}:
  1.0
  0.5
 -0.5
 -1.0
```

See also [`chebdiff`](@ref), [`chebws`](@ref).
"""
chebpts(N::Int, ::Type{T}=Float64) where {T} = T.(cos.((0:(N - 1))π/(N - 1)))

end
