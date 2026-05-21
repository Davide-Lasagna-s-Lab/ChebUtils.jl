# Clenshaw–Curtis quadrature weights for the Chebyshev–Gauss–Lobatto grid.
#
# Original MATLAB code by Professor Lloyd N. Trefethen (Oxford University).
# Source: http://people.maths.ox.ac.uk/trefethen/clencurt.m
# Adapted to Julia with minor style changes.
#
# Theory
# ──────
# On the N-point CGL grid xⱼ = cos(πj/n), j = 0,…,n (n = N-1), the
# Clenshaw–Curtis rule integrates exactly every polynomial of degree ≤ n.
# The weights are derived by integrating the trigonometric interpolant of f
# on [0, π] and transforming back to [-1, 1].
#
# For smooth (C^∞) functions the error decreases geometrically in N.

"""
    chebws(N::Int) -> Vector{Float64}

Return the `N` Clenshaw–Curtis quadrature weights for the N-point
Chebyshev–Gauss–Lobatto (CGL) grid on [-1, 1].

The weights `w` satisfy

    ∫₋₁¹ f(x) dx  ≈  ∑ⱼ wⱼ · f(xⱼ)

with `xⱼ = cos(π(j-1)/(N-1))`, j = 1,…,N.

The quadrature rule is exact for all polynomials of degree ≤ N-1 and
converges geometrically for smooth functions.

# Algorithm

The computation follows Trefethen (2000, Program 12):

1. Endpoints receive weight `1/(n²−1)` (n even) or `1/n²` (n odd),
   where `n = N−1`.
2. Interior weights are assembled in DCT-like fashion by accumulating cosine
   corrections in a vector `v`, then scaling by `2/n`.

# Examples
```julia
julia> y  = chebpts(16);
julia> ws = chebws(16);
julia> abs(sum(ws .* exp.(y)) - (exp(1) - exp(-1))) < 1e-12
true

julia> # Exact integration of x³ on [-1,1] (antisymmetric → zero)
julia> y2 = chebpts(3); ws2 = chebws(3);
julia> abs(sum(ws2 .* y2.^3)) < 1e-15
true
```

See also [`chebpts`](@ref).

# References
- Trefethen, L.N. (2000). *Spectral Methods in MATLAB*, Program 12. SIAM.
"""
function chebws(N::Int)
    n = N - 1                     # polynomial degree = number of intervals
    θ = (0:n) * π / n             # CGL angles: θⱼ = πj/n

    ws = zeros(n + 1)
    v  = ones(n - 1)              # accumulator for interior weights

    if mod(n, 2) == 0
        # ── Even n ──────────────────────────────────────────────────────────
        # Endpoint weights (Trefethen eq. 12.3)
        ws[1] = ws[n + 1] = 1/(n^2 - 1)

        # Interior: subtract even-frequency cosine terms up to Nyquist-1
        for k in 1:(n/2 - 1)
            v .-= 2*cos.(2*k*θ[2:n]) / (4k^2 - 1)
        end

        # Nyquist term (k = n/2) has a different prefactor
        v .-= cos.(n*θ[2:n]) / (n^2 - 1)
    else
        # ── Odd n ───────────────────────────────────────────────────────────
        ws[1] = ws[n + 1] = 1/n^2

        for k in 1:((n - 1)/2)
            v .-= 2*cos.(2*k*θ[2:n]) / (4k^2 - 1)
        end
    end

    ws[2:n] .= 2*v / n            # scale interior weights

    return ws
end

# ── Legacy / alternative implementation ─────────────────────────────────────
# chebws_dep computes weights via an inversion of the (N-1)×(N-1) leading
# sub-block of the differentiation matrix.  This is exact but O(N³) and
# kept only for cross-validation purposes.  Use chebws for production code.
chebws_dep(D::ChebDiff{T}) where {T} = append!(inv(D[1:(end - 1), 1:(end - 1)])[1, :], [0])
chebws_dep(N::Int) = chebws_dep(chebdiff(N))
