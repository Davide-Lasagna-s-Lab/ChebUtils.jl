# Definition of AdjointChebDiff and its construction via adjoint(D) / adjoint(D, w).
#
# The Chebyshev differentiation matrix D is defined under the standard Euclidean
# inner product, but spectral operators often live inside a weighted space with
# inner product
#
#     ⟨u, v⟩_W = ∑ⱼ wⱼ u(xⱼ) v(xⱼ)  =  u' W v,    W = Diagonal(w)
#
# where w are the Clenshaw-Curtis weights from chebws.  In that weighted space,
# the adjoint of D is the operator D† satisfying ⟨u, D v⟩_W = ⟨D† u, v⟩_W,
# which works out to:
#
#     D† = W⁻¹ D' W,    (D†)ᵢⱼ = Dⱼᵢ · wⱼ / wᵢ
#
# The standard (Euclidean) adjoint is the special case w = ones(N), giving D† = D'.
#
# AdjointChebDiff precomputes the adjoint matrix once at construction time so that
# all subsequent mul! calls are allocation-free, exactly mirroring the pattern used
# by AdjointDiffMatrix in FDGrids.jl.

"""
    AdjointChebDiff{T<:Real, N} <: AbstractMatrix{T}

The adjoint (transpose, or weighted adjoint) of an N×N Chebyshev differentiation
matrix, constructed from a `ChebDiff` via `adjoint`.

Two construction forms are supported:

- `adjoint(D)` — the standard (Euclidean) adjoint, i.e. the matrix transpose `D'`.
- `adjoint(D, w)` — the weighted adjoint `W⁻¹ D' W` where `W = Diagonal(w)`.

The adjoint matrix is precomputed once at construction time. All subsequent
`mul!` calls operate on this stored matrix with no allocations.

`adjoint(A::AdjointChebDiff)` returns the original `ChebDiff` without copying.

# Notes

- `setindex!` is not supported; mutating `AdjointChebDiff` directly would leave
  the precomputed matrix stale.  Modify the parent and reconstruct instead.
- For the Clenshaw-Curtis inner product, pass `w = chebws(N)` to get the
  spectrally correct weighted adjoint.

# Examples
```julia
julia> N  = 32;
julia> D  = chebdiff(N);
julia> Dt = adjoint(D)
32×32 AdjointChebDiff{Float64, 32}

julia> # Weighted adjoint under the Clenshaw-Curtis inner product
julia> w  = chebws(N);
julia> Dw = adjoint(D, w)
32×32 AdjointChebDiff{Float64, 32}

julia> # Round-trip: adjoint of adjoint returns the parent
julia> adjoint(Dt) === D
true
```

See also [`ChebDiff`](@ref), [`chebdiff`](@ref), [`chebws`](@ref).
"""
struct AdjointChebDiff{T<:Real, N} <: AbstractMatrix{T}
    parent :: ChebDiff{T, N}
    mat    :: Matrix{T}       # precomputed adjoint (possibly weighted) matrix
end


# ── Construction ──────────────────────────────────────────────────────────────

"""
    adjoint(D::ChebDiff{T}) -> AdjointChebDiff{T,N}

Construct the Euclidean adjoint (matrix transpose) of the Chebyshev
differentiation matrix `D`.

The transposed matrix `D'` is precomputed once and stored inside the returned
`AdjointChebDiff`.  All subsequent `mul!` calls are allocation-free.

# Examples
```julia
julia> D  = chebdiff(16);
julia> Dt = adjoint(D)

julia> # Verify adjoint identity: ⟨v, D w⟩ = ⟨Dᵀ v, w⟩
julia> v, w = randn(16), randn(16);
julia> v' * (D * w) ≈ (Dt * v)' * w
true
```

See also [`adjoint(::ChebDiff, ::AbstractVector)`](@ref).
"""
function LinearAlgebra.adjoint(D::ChebDiff{T}) where {T}
    return AdjointChebDiff(D, collect(transpose(D.mat)))
end

"""
    adjoint(D::ChebDiff{T}, w::AbstractVector{T}) -> AdjointChebDiff{T,N}

Construct the weighted adjoint `D† = W⁻¹ D' W` of the Chebyshev differentiation
matrix `D` under the diagonal inner product `W = Diagonal(w)`.

The entry `(D†)ᵢⱼ = Dⱼᵢ · wⱼ / wᵢ` is precomputed at construction time.

# Arguments
- `D` : first-order (or second-order) Chebyshev differentiation matrix.
- `w` : quadrature weight vector of length `N`.  For the spectrally consistent
        Clenshaw-Curtis inner product, pass `chebws(N)`.  All weights must be
        strictly positive.

# Examples
```julia
julia> N  = 32;
julia> D  = chebdiff(N);
julia> w  = chebws(N);
julia> Dw = adjoint(D, w)

julia> # Weighted adjoint identity: ⟨v, D u⟩_W = ⟨D† v, u⟩_W
julia> u, v = randn(N), randn(N);
julia> dot(v .* w, D * u) ≈ dot(Dw * v .* w, u)
true
```
"""
function LinearAlgebra.adjoint(D::ChebDiff{T}, w::AbstractVector{T}) where {T}
    N = size(D, 1)
    length(w) == N ||
        throw(ArgumentError("length(w) = $(length(w)) must equal size(D,1) = $N"))
    all(>(0), w) ||
        throw(ArgumentError("all weights must be strictly positive"))

    # (D†)ᵢⱼ = Dⱼᵢ · wⱼ / wᵢ  —  fold weights into the transposed matrix
    mat = [D[j, i] * w[j] / w[i] for i in 1:N, j in 1:N]
    return AdjointChebDiff(D, mat)
end

"""
    adjoint(A::AdjointChebDiff) -> ChebDiff

Return the parent forward matrix.  No copy is made.

This is a structural unwrap: `AdjointChebDiff` retains a reference to the
`ChebDiff` from which it was built, and this method returns that object directly.
For a weighted adjoint `adjoint(D, w)`, this returns `D` (not the standard
Euclidean adjoint of `W⁻¹ D' W`).
"""
LinearAlgebra.adjoint(A::AdjointChebDiff) = A.parent


# ── AbstractMatrix interface ──────────────────────────────────────────────────

Base.size(::AdjointChebDiff{T, N}) where {T, N} = (N, N)
Base.IndexStyle(::Type{<:AdjointChebDiff}) = Base.IndexLinear()

Base.@propagate_inbounds function Base.getindex(A::AdjointChebDiff, I...)
    @boundscheck checkbounds(A.mat, I...)
    @inbounds return A.mat[I...]
end

"""
    setindex!(::AdjointChebDiff, v, I...)

Reject in-place mutation of `AdjointChebDiff`.

The adjoint matrix is precomputed from the parent `ChebDiff`.  Mutating it
directly would leave the stored coefficients inconsistent with the parent.
To change the adjoint, modify the parent matrix and call `adjoint` again.
"""
function Base.setindex!(::AdjointChebDiff, v, I...)
    throw(ArgumentError(
        "setindex! is not supported on AdjointChebDiff: modifying it in-place " *
        "would leave the precomputed matrix stale. " *
        "Modify the parent ChebDiff and reconstruct via adjoint(D)."))
end

Base.copy(A::AdjointChebDiff) = AdjointChebDiff(A.parent, copy(A.mat))

"""
    parent(A::AdjointChebDiff) -> Matrix

Return the underlying precomputed dense adjoint matrix without copying.
"""
Base.parent(A::AdjointChebDiff) = A.mat
