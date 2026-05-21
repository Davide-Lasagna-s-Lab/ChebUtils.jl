# Factory functions that build Chebyshev differentiation matrices.
#
# Both chebdiff and chebddiff use the Chebyshev–Gauss–Lobatto (CGL) grid:
#
#     xⱼ = cos(π(j-1)/(N-1)),   j = 1, …, N
#
# and follow the formulas from Trefethen (2000), Chapter 6.

"""
    chebdiff(N, T=Float64) -> ChebDiff{T,N}

Construct the N×N first-order Chebyshev spectral differentiation matrix `D`
on the Chebyshev–Gauss–Lobatto (CGL) grid.

Given a vector of function values `f = [f(x₁), …, f(xₙ)]` at the CGL points,
`D * f` returns a spectral approximation of the first derivative `f'` at
the same points.

# Mathematical definition

For i ≠ j:

    Dᵢⱼ = (cᵢ/cⱼ) · (-1)^(i+j) / (xᵢ - xⱼ)

where `cᵢ = 2` if i = 1 or i = N, and `cᵢ = 1` otherwise.

Diagonal entries are set by the "negative-sum trick" to ensure that constant
functions are differentiated to zero:

    Dᵢᵢ = −∑_{j≠i} Dᵢⱼ

# Arguments
- `N::Int`  : number of grid points.
- `T::Type` : element type of the output matrix (default `Float64`).
              When `T <: Real` the result is a `ChebDiff{T,N}`.
              For non-real types (e.g. `ComplexF64`) a plain `Matrix{T}` is
              returned instead.

# Examples
```julia
julia> D = chebdiff(4)
4×4 ChebDiff{Float64, 4}: …

julia> y = chebpts(4);
julia> D * sin.(y)   # spectral derivative of sin
```

See also [`chebddiff`](@ref), [`chebpts`](@ref).
"""
function chebdiff(N::Int, ::Type{T}=Float64) where {T}
    D = zeros(T, N, N)

    # cₖ = 2 at the endpoints (k=1 or k=N), 1 elsewhere
    c = i -> i == 1 || i == N ? 2 : 1

    # CGL node positions
    x = i -> cos(((i - 1)*π)/(N - 1))

    # Off-diagonal entries via the Trefethen formula
    for i in 1:N, j in 1:N
        if i != j
            D[i, j] = (c(i)/c(j))*(((-1)^(i + j))/(x(i) - x(j)))
        end
    end

    # Diagonal entries: negative-sum trick (exact for constant functions)
    for i in 1:N
        D[i, i] = -sum([D[i, j] for j in 1:N if i != j])
    end

    return ChebDiff(D, T)
end

"""
    chebddiff(D::ChebDiff{T}) -> ChebDiff{T,N}
    chebddiff(N::Int, T=Float64) -> ChebDiff{T,N}

Construct the N×N second-order Chebyshev spectral differentiation matrix `DD`
on the Chebyshev–Gauss–Lobatto (CGL) grid.

`DD * f` approximates the second derivative `f''` at the CGL nodes.

# Mathematical definition

The entries are computed from the first-order matrix `D` via the Fornberg (1998)
recurrence, which is more numerically stable than squaring `D` directly:

For i ≠ j:

    DDᵢⱼ = 2·Dᵢⱼ·(Dᵢᵢ − 1/(xᵢ − xⱼ))

For diagonal entries:

    DDᵢᵢ = 2·(Dᵢᵢ² + ∑_{k≠i} Dᵢₖ/(xᵢ − xₖ))

# Arguments
- `D::ChebDiff{T}` : a pre-computed first-order matrix from [`chebdiff`](@ref).
- `N::Int` / `T`   : convenience overload that calls `chebdiff(N, T)` first.

# Notes
The two-argument form `chebddiff(N, T)` is a convenience wrapper.  When you
already have `D` (e.g. for solving a BVP that needs both first and second
derivatives), pass it directly to avoid redundant computation.

# Examples
```julia
julia> DD = chebddiff(32)      # second-derivative matrix on 32 CGL points
julia> y  = chebpts(32)
julia> DD * exp.(y) ≈ exp.(y)  # true (exp is its own second derivative)
true
```

See also [`chebdiff`](@ref).
"""
function chebddiff(D::ChebDiff{T}) where {T}
    DD = zero(D)
    N  = size(D, 1)

    x = i -> cos(((i - 1)*π)/(N - 1))

    for i in 1:N, j in 1:N
        if i != j
            DD[i, j] = 2*D[i, j]*(D[i, i] - 1/(x(i) - x(j)))
        else
            DD[i, i] = 2*(D[i, i]^2 + sum([D[i, k]/(x(i) - x(k)) for k in 1:N if k != i]))
        end
    end

    return DD
end

# Convenience overload: build D internally, then derive DD.
chebddiff(N::Int, ::Type{T}=Float64) where {T} = chebddiff(chebdiff(N, T))
