# This file defines how the Linear Algebra multiply method works, providing a
# convenient method of differentiating fields of various dimensions.
#
# Both ChebDiff (forward) and AdjointChebDiff (adjoint / weighted adjoint)
# share the same generated loop structure.  A single Union pair covers both.
#
# The 1-D method delegates directly to parent(D) (the underlying dense Matrix).
# The N-D @generated version builds a bespoke nested loop at compile time that
# slices along dimension DIM and calls the 1-D method on each fibre.

# Union alias so the two methods below cover both operator types.
const ChebOp{T} = Union{ChebDiff{T}, AdjointChebDiff{T}}

# 1-D: delegate to the underlying dense Matrix stored in parent(D).
function LinearAlgebra.mul!(y::AbstractArray{S, 1},
                            D::ChebOp{T},
                            x::AbstractArray{S, 1}) where {S, T}
    LinearAlgebra.mul!(y, parent(D), x)
end

# N-D: differentiate array x along dimension DIM, storing result in y.
# The @generated macro specialises the loop nest at compile time for every
# concrete (S, N, T, DIM) tuple, so the compiler sees plain indexing with
# no overhead from the dimension-selection logic.
@generated function LinearAlgebra.mul!(y::AbstractArray{S, N},
                                       D::ChebOp{T},
                                       x::AbstractArray{S, N},
                                        ::Val{DIM}=Val(1)) where {S, N, T, DIM}
    var(d)  = Symbol(:i_, d)
    x_slice = Expr(:ref, :x, ntuple(i -> i == DIM ? :(:) : var(i), N)...)
    y_slice = Expr(:ref, :y, ntuple(i -> i == DIM ? :(:) : var(i), N)...)
    Ni = [Symbol(:N, d) for d in 1:N]
    loop_head = [Expr(:(=), var(d), Expr(:call, :(:), 1, Symbol(:N, d))) for d in N:-1:1 if d != DIM]
    loop_expr = Expr(:for, Expr(:block, loop_head...), :(mul!($y_slice, D, $x_slice)))
    return quote
        $(Expr(:(=), Expr(:tuple, Ni...), :(size(y))))
        @inbounds @views begin
            $loop_expr
        end
        return y
    end
end
