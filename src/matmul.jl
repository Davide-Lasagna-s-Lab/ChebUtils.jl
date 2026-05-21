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

# 1-D specialisation: thin wrapper around the underlying dense matrix multiply.
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

# 1-D specialisation: multiply by the precomputed adjoint matrix stored in A.mat.
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
