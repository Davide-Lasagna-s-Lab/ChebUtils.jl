# Clenshaw-Curtis quadrature weights for the Chebyshev-Gauss-Lobatto grid.
#
# Original algorithm by Lloyd N. Trefethen (Oxford).
# Adapted from http://people.maths.ox.ac.uk/trefethen/clencurt.m

"""
    chebws(N::Int) -> Vector{Float64}

Compute the `N`-point Clenshaw-Curtis quadrature weights for the
Chebyshev–Gauss–Lobatto (CGL) grid returned by [`chebpts`](@ref).

The weights `w` satisfy

```math
\\int_{-1}^{1} f(x)\\, dx \\approx \\sum_{j=1}^{N} w_j\\, f(x_j)
```

with spectral (exponential) convergence for smooth `f`.  For polynomial `f`
of degree ≤ `N−1` the rule is exact.

The boundary weights are `w₁ = wₙ = 1/(N²−1)` (even `N−1`) or `1/(N−1)²`
(odd `N−1`); interior weights are computed by the DCT-based Trefethen algorithm.

# Examples
```julia
julia> w = chebws(16);
julia> y = chebpts(16);
julia> abs(sum(w .* exp.(y)) - (exp(1) - exp(-1))) < 1e-12
true
```

See also [`chebpts`](@ref).
"""
function chebws(N::Int)
    # obtain domain information
    N -= 1
    θ = (0:N)π/N

    # intialise vectors
    ws = zeros(N + 1)
    v = ones(N - 1)

    # check if N is even or odd
    if mod(N, 2) == 0
        # compute end weights
        ws[1] = ws[N + 1] = 1/(N^2 - 1)

        # loop over interior points updating v
        for k in 1:(N/2 - 1)
            v .-= 2*cos.(2*k*θ[2:N])/(4*(k^2) - 1)
        end

        v .-= cos.(N*θ[2:N])/(N^2 - 1)
    else
        # compute end points
        ws[1] = ws[N + 1] = 1/N^2

        # loop over interior points updating v
        for k in 1:((N - 1)/2)
            v .-= 2*cos.(2*k*θ[2:N])/(4*(k^2) - 1)
        end
    end

    # assign interior values for weights
    ws[2:N] .= 2*v/N

    return ws
end

chebws_dep(D::ChebDiff{T}) where {T} = append!(inv(D[1:(end - 1), 1:(end - 1)])[1, :], [0])
chebws_dep(N::Int) = chebws_dep(chebdiff(N))