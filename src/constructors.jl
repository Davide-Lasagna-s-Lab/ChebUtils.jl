# Constructors for ChebDiff and the second-order matrix chebddiff.
#
# chebdiff  — first-order N×N spectral differentiation matrix on the CGL grid.
# chebddiff — second-order matrix, either directly via the closed-form formula
#             or as the square of a given ChebDiff.

"""
    chebdiff(N::Int, T::Type=Float64) -> ChebDiff{T,N}

Construct the `N×N` first-order Chebyshev spectral differentiation matrix on
the Chebyshev–Gauss–Lobatto (CGL) grid

```
xⱼ = cos(π(j−1)/(N−1)),   j = 1, …, N
```

so that `(chebdiff(N) * f)[i] ≈ f'(xᵢ)` for any smooth function `f`.

The off-diagonal entries follow the Fornberg/Trefethen formula

```
Dᵢⱼ = (cᵢ/cⱼ) · (−1)^(i+j) / (xᵢ − xⱼ),   i ≠ j
```

where `cⱼ = 2` for `j = 1` or `j = N` and `cⱼ = 1` otherwise.  Diagonal
entries are set by the negative-sum trick to ensure exact differentiation of
constants.

# Examples
```julia
julia> D = chebdiff(4)
4×4 ChebDiff{Float64, 4}

julia> D ≈ [19/6 -4.0 4/3 -0.5; 1.0 -1/3 -1.0 1/3;
            -1/3  1.0 1/3 -1.0; 0.5 -4/3 4.0 -19/6]
true
```

See also [`chebddiff`](@ref), [`chebpts`](@ref).
"""
function chebdiff(N::Int, ::Type{T}=Float64) where {T}
    # initialise matrix
    D = zeros(T, N, N)

    # define anonymous function for c_k coefficient and x location
    c = i -> i == 1 || i == N ? 2 : 1
    x = i -> cos(((i - 1)*π)/(N - 1))

    # loop over matrix assigning values
    for i in 1:N, j in 1:N
        if i != j
            D[i, j] = (c(i)/c(j))*(((-1)^(i + j))/(x(i) - x(j)))
        end
    end

    # compute diagonal entries
    for i in 1:N
        D[i, i] = -sum([D[i, j] for j in 1:N if i != j])
    end

    return ChebDiff(D, T)
end

"""
    chebddiff(N::Int, T::Type=Float64) -> ChebDiff{T,N}
    chebddiff(D::ChebDiff) -> ChebDiff

Construct the `N×N` second-order Chebyshev spectral differentiation matrix,
so that `(chebddiff(N) * f)[i] ≈ f''(xᵢ)`.

When called with a `ChebDiff` `D`, the closed-form formula is applied directly
to `D` — this is more numerically stable than squaring `D` explicitly.  The
two-argument form `chebddiff(N)` constructs `D` internally and applies the same
formula.

# Examples
```julia
julia> N  = 32;
julia> D  = chebdiff(N);
julia> DD = chebddiff(D)    # closed-form second derivative
julia> DD.mat ≈ D.mat * D.mat   # equivalent to squaring for smooth grids
true
```

See also [`chebdiff`](@ref).
"""
function chebddiff(D::ChebDiff{T}) where {T}
    # initialise matrix
    DD = zero(D)

    # define anonymous function for c_k coefficients and x location
    x = i -> cos(((i - 1)*π)/(size(D)[1] - 1))

    # loop over matrix assigning values
    for i in 1:size(D)[1], j in 1:size(D)[1]
        if i != j
            DD[i, j] = 2*D[i, j]*(D[i, i] - 1/(x(i) - x(j)))
        else
            DD[i, i] = 2*(D[i, i]^2 + sum([D[i, k]/(x(i) - x(k)) for k in 1:size(D)[1] if k != i]))
        end
    end

    return DD
end
chebddiff(N::Int, ::Type{T}=Float64) where {T} = chebddiff(chebdiff(N, T))
