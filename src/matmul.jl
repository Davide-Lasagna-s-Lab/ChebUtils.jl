# This file defines how the Linear Algebra multiply method works, providing a
# convenient method of differentiating fields of various dimensions.

# differentiate a vector, x
function LinearAlgebra.mul!(y::AbstractArray{S, 1},
                            D::ChebDiff{T},
                            x::AbstractArray{S, 1}) where {S, T}
    LinearAlgebra.mul!(y, D.mat, x)
end

# differentiate an arbitrary dimension array, x, along the first direction
function LinearAlgebra.mul!(y::AbstractArray{S, ND},
                            D::ChebDiff{T},
                            x::AbstractArray{S, ND}) where {S, ND, T}
    @views @inbounds begin
        for I in CartesianIndices(size(x)[2:end])
            LinearAlgebra.mul!(y[:, I], D, x[:, I])
        end
    end

    return y
end

# differentiate an arbitrary dimension array, x, along a given direction, dir.
function LinearAlgebra.mul!(y::AbstractArray{S, N},
                            D::ChebDiff{T},
                            x::AbstractArray{S, N},
                            dir::Int) where {S, N, T}
    dims = filter(i->i!=dir, ntuple(i->i, N)) # FIXME: this costs a bunch of memory
    for (u, v) in zip(eachslice(x, dims=dims), eachslice(y, dims=dims))
        mul!(v, D, u)
    end

    return y
end
