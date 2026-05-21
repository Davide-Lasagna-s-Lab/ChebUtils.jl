# Definition of the ChebDiff matrix type and its AbstractMatrix interface.
#
# ChebDiff wraps a plain dense Matrix but carries the grid size N as a type
# parameter.  This lets dispatch select specialised mul! kernels without any
# runtime overhead, while still integrating transparently with everything in
# LinearAlgebra (lu!, broadcasting, …).

"""
    ChebDiff{T<:Real, N} <: AbstractMatrix{T}

A Chebyshev differentiation matrix of size N×N with element type `T`.

`ChebDiff` behaves like an ordinary `AbstractMatrix`: you can index it, copy
it, broadcast over it, and pass it to any `LinearAlgebra` routine.  Internally
it stores a plain `Matrix{T}`, which can be retrieved with `parent`.

The type parameter `N` is the grid size and is baked in at construction time so
that `@generated` dispatch (e.g. [`mul!`](@ref)) can resolve dimension loops at
compile time with no runtime overhead.

# Construction

Do not construct `ChebDiff` directly.  Use the factory functions instead:

- [`chebdiff(N)`](@ref)  – first-order differentiation matrix
- [`chebddiff(N)`](@ref) – second-order differentiation matrix

Both accept an optional precision type argument (`Float32`, `BigFloat`, …).

# Notes

- When the requested element type is not a subtype of `Real` (e.g. `ComplexF64`)
  the factory functions return a plain `Matrix{T}` rather than a `ChebDiff`.
  This is deliberate: complex-valued differentiation matrices are unusual and
  the wrapper provides no benefit beyond the real case.

# Examples
```julia
julia> D = chebdiff(4)
4×4 ChebDiff{Float64, 4}:
  3.16667  -4.0      1.33333  -0.5
  1.0      -0.33333  -1.0      0.33333
 -0.33333   1.0      0.33333  -1.0
  0.5      -1.33333   4.0     -3.16667
```

See also [`chebdiff`](@ref), [`chebddiff`](@ref).
"""
struct ChebDiff{T<:Real, N} <: AbstractMatrix{T}
    mat::Matrix{T}

    ChebDiff(mat) = new{eltype(mat), size(mat, 1)}(mat)
end

# Convenience constructor: convert element type to S when S<:Real,
# or fall back to a plain Matrix{S} for non-real types (e.g. Complex).
ChebDiff(mat::AbstractMatrix, ::Type{S}) where {S<:Real} = ChebDiff(S.(mat))
ChebDiff(mat::AbstractMatrix, ::Type{S}) where {S}       = Matrix{S}(S.(mat))

# ── AbstractMatrix interface ────────────────────────────────────────────────

Base.size(::ChebDiff{T, N}) where {T, N} = (N, N)

# Linear indexing is fastest because the backing store is a plain Matrix.
Base.IndexStyle(::Type{<:ChebDiff}) = Base.IndexLinear()

"""
    parent(D::ChebDiff) -> Matrix

Return the underlying dense matrix without copying.
Useful when you need to pass the raw data to a routine that does not accept
`AbstractMatrix`.
"""
Base.parent(D::ChebDiff) = D.mat

Base.similar(D::ChebDiff{T}, ::Type{S}=T) where {T, S} = ChebDiff(similar(parent(D)), S)
Base.copy(D::ChebDiff)                                  = ChebDiff(copy(parent(D)))

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

# ── Broadcasting ─────────────────────────────────────────────────────────────
# Broadcasting style follows the pattern from MultiscaleArrays.jl:
# operations that produce an array of the same shape stay wrapped in ChebDiff.

const ChebDiffStyle = Broadcast.ArrayStyle{ChebDiff}
Base.BroadcastStyle(::Type{<:ChebDiff}) = Broadcast.ArrayStyle{ChebDiff}()

# Allocate the output buffer for a broadcast expression involving ChebDiff.
Base.similar(bc::Base.Broadcast.Broadcasted{ChebDiffStyle}, ::Type{T}) where {T} =
    ChebDiff(similar(Array{Float64}, axes(bc)), T)
