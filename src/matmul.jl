# This file defines how the Linear Algebra multiply method works, providing a
# convenient method of differentiating fields of various dimensions.
#
# Both ChebDiff (forward) and AdjointChebDiff (adjoint / weighted adjoint)
# share the same generated loop structure.  The 1-D specialisation delegates
# directly to the underlying dense Matrix multiply; the N-D @generated version
# builds a bespoke nested loop at compile time that slices along dimension DIM
# and calls the 1-D method on each fibre.

const __VARS__ = (:i, :j, :k, :l)

# ── ChebDiff (forward) ────────────────────────────────────────────────────────

"""
    mul!(y, D::ChebDiff, x) -> y
    mul!(y, D::ChebDiff, x, Val(DIM)) -> y

In-place Chebyshev spectral differentiation of `x`, storing the result in `y`.

**1-D form** (`x` and `y` are vectors): computes `y = D * x` via the
underlying dense matrix multiply.

**N-D form** (`x` and `y` are N-dimensional arrays, `DIM` is a `Val`):
differentiates `x` along dimension `DIM`, applying `D` to each 1-D fibre
`x[..., :, ...]`.  The loop nest is generated at compile time — no views,
no closures, no runtime dispatch.

# Examples
```julia
julia> N = 32; y = chebpts(N); D = chebdiff(N);
julia> f  = exp.(y);                 # f(y) = eʸ
julia> df = similar(f);
julia> mul!(df, D, f)                # df ≈ eʸ = f
julia> df ≈ f
true

julia> # 3-D field: differentiate along dimension 2
julia> F  = randn(8, N, 8);
julia> dF = similar(F);
julia> mul!(dF, D, F, Val(2))
```

See also [`mul!(y, AdjointChebDiff, x)`](@ref).
"""
function LinearAlgebra.mul!(y::AbstractArray{S, 1},
                            D::ChebDiff{T},
                            x::AbstractArray{S, 1}) where {S, T}
    LinearAlgebra.mul!(y, D.mat, x)
end

# N-D: differentiate array x along dimension DIM, storing result in y.
# The @generated macro builds a loop nest at compile time for every concrete
# (S, N, T, DIM) so the compiler sees plain indexing loops with no overhead.
@generated function LinearAlgebra.mul!(y::AbstractArray{S, N},
                                       D::ChebDiff{T},
                                       x::AbstractArray{S, N},
                                        ::Val{DIM}=Val(1)) where {S, N, T, DIM}
    x_slice = Expr(:ref, :x, ntuple(i->i==DIM ? :(:) : __VARS__[i], N)...)
    y_slice = Expr(:ref, :y, ntuple(i->i==DIM ? :(:) : __VARS__[i], N)...)
    Ni = [Symbol(:N, d) for d in 1:N]
    loop_head = [Expr(:(=), __VARS__[d], Expr(:call, :(:), 1, Symbol(:N, d))) for d in N:-1:1 if d != DIM]
    loop_expr = Expr(:for, Expr(:block, loop_head...), :(mul!($y_slice, D, $x_slice)))
    return quote
        $(Expr(:(=), Expr(:tuple, Ni...), :(size(y))))
        @inbounds @views begin
            $loop_expr
        end
        return y
    end
end


# ── AdjointChebDiff ───────────────────────────────────────────────────────────

"""
    mul!(y, A::AdjointChebDiff, x) -> y
    mul!(y, A::AdjointChebDiff, x, Val(DIM)) -> y

In-place multiplication by the precomputed (possibly weighted) adjoint matrix.

Identical calling convention to [`mul!(y, ::ChebDiff, x)`](@ref): the 1-D
form wraps a dense matrix multiply against the stored adjoint matrix; the N-D
form differentiates along `DIM`.

# Examples
```julia
julia> N = 32; D = chebdiff(N); Dt = adjoint(D);
julia> v = randn(N); w = similar(v);
julia> mul!(w, Dt, v)            # w = Dᵀ v
julia> w ≈ Matrix(D)' * v
true
```
"""
function LinearAlgebra.mul!(y::AbstractArray{S, 1},
                            A::AdjointChebDiff{T},
                            x::AbstractArray{S, 1}) where {S, T}
    LinearAlgebra.mul!(y, A.mat, x)
end

# N-D: same generated loop structure as for ChebDiff, but slices call the
# AdjointChebDiff 1-D method above.
@generated function LinearAlgebra.mul!(y::AbstractArray{S, N},
                                       A::AdjointChebDiff{T},
                                       x::AbstractArray{S, N},
                                        ::Val{DIM}=Val(1)) where {S, N, T, DIM}
    x_slice = Expr(:ref, :x, ntuple(i->i==DIM ? :(:) : __VARS__[i], N)...)
    y_slice = Expr(:ref, :y, ntuple(i->i==DIM ? :(:) : __VARS__[i], N)...)
    Ni = [Symbol(:N, d) for d in 1:N]
    loop_head = [Expr(:(=), __VARS__[d], Expr(:call, :(:), 1, Symbol(:N, d))) for d in N:-1:1 if d != DIM]
    loop_expr = Expr(:for, Expr(:block, loop_head...), :(mul!($y_slice, A, $x_slice)))
    return quote
        $(Expr(:(=), Expr(:tuple, Ni...), :(size(y))))
        @inbounds @views begin
            $loop_expr
        end
        return y
    end
end
