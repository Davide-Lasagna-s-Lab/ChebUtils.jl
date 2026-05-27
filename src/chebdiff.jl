# Chebyshev differentiation matrix type and its AbstractMatrix interface.
#
# ChebDiff{T,N} wraps an N×N dense matrix whose (i,j) entry is the spectral
# weight that maps the value of a function at CGL node j to its derivative at
# CGL node i.  The grid runs from +1 (row/col 1) to −1 (row/col N), following
# the Trefethen convention.  The matrix is computed once and reused; all
# differentiation is a single dense matrix–vector multiply.

"""
    ChebDiff{T<:Real, N} <: AbstractMatrix{T}

An `N×N` Chebyshev spectral differentiation matrix for the Chebyshev–Gauss–Lobatto
(CGL) grid on `[-1, 1]`.

Indexing row `i`, column `j` gives the weight that maps the function value at
CGL node `j` to the derivative at node `i`.  The grid runs from `+1` (index 1)
to `−1` (index `N`) following the Trefethen convention.

Construct via [`chebdiff`](@ref) (first order) or [`chebddiff`](@ref) (second
order).  Apply to a vector or multi-dimensional array via `mul!`.

`ChebDiff` implements the full `AbstractMatrix` interface (size, getindex,
setindex!, broadcasting) and supports in-place LU factorisation via `lu!`.
"""
struct ChebDiff{T<:Real, N} <: AbstractMatrix{T}
    mat::Matrix{T}

    ChebDiff(mat) = new{eltype(mat), size(mat, 1)}(mat)
end

ChebDiff(mat::AbstractMatrix, ::Type{S}) where {S<:Real} = ChebDiff(S.(mat))
ChebDiff(mat::AbstractMatrix, ::Type{S}) where {S} = Matrix{S}(S.(mat))

Base.size(::ChebDiff{T, N}) where {T, N} = (N, N)
Base.IndexStyle(::Type{<:ChebDiff}) = Base.IndexLinear()
Base.parent(D::ChebDiff) = D.mat
Base.similar(D::ChebDiff{T}, ::Type{S}=T) where {T, S} = ChebDiff(similar(parent(D)), S)
Base.copy(D::ChebDiff) = ChebDiff(copy(parent(D)))

Base.@propagate_inbounds function Base.getindex(D::ChebDiff, I...)
    @boundscheck checkbounds(parent(D), I...)
    @inbounds ret = parent(D)[I...]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(D::ChebDiff, v, I...)
    @boundscheck checkbounds(parent(D), I...)
    @inbounds parent(D)[I...] = v
    return v
end

# ~ BROADCASTING ~
# taken from MultiscaleArrays.jl
const ChebDiffStyle = Broadcast.ArrayStyle{ChebDiff}
Base.BroadcastStyle(::Type{<:ChebDiff}) = Broadcast.ArrayStyle{ChebDiff}()

# for broadcasting to construct new objects
Base.similar(bc::Base.Broadcast.Broadcasted{ChebDiffStyle}, ::Type{T}) where {T} = ChebDiff(similar(Array{Float64}, axes(bc)), T)
