# This file defines how the Linear Algebra multiply method works, providing a
# convenient method of differentiating fields of various dimensions.

const __VARS__ = (:i, :j, :k, :l)

# differentiate a vector, x
function LinearAlgebra.mul!(y::AbstractArray{S, 1},
                            D::ChebDiff{T},
                            x::AbstractArray{S, 1}) where {S, T}
    LinearAlgebra.mul!(y, D.mat, x)
end

# differentiate an arbitrary dimension array, x, along a given direction, dir.
@generated function LinearAlgebra.mul!(y::AbstractArray{S, N},
                                       D::ChebDiff{T},
                                       x::AbstractArray{S, N},
                                        ::Val{DIM}=Val(1)) where {S, N, T, DIM}
    # generate slice expressions
    x_slice = Expr(:ref, :x, ntuple(i->i==DIM ? :(:) : __VARS__[i], N)...)
    y_slice = Expr(:ref, :y, ntuple(i->i==DIM ? :(:) : __VARS__[i], N)...)

    # define N1, N2, N3, N4
    Ni = [Symbol(:N, d) for d in 1:N]

    # create loop expression
    loop_head = [Expr(:(=), __VARS__[d], Expr(:call, :(:), 1, Symbol(:N, d))) for d in N:-1:1 if d != DIM]
    loop_expr = Expr(:for, Expr(:block, loop_head...), :(mul!($y_slice, D, $x_slice)))

    output = quote
        # size of array as a tuple
        # e.g. N1, N2, N3 = size(y)
        $(Expr(:(=), Expr(:tuple, Ni...), :(size(y))))

        @inbounds @views begin
            $loop_expr
        end

        return y
    end

    return output
end
