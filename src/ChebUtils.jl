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
