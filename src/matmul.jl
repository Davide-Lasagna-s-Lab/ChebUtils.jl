# Specialised mul! dispatch for ChebDiff.
#
# The key design goal is to differentiate a field stored in an N-dimensional
# array along a single chosen dimension without any allocations.  The
# @generated macro lets us build a bespoke loop nest at compile time so the
# compiler sees plain indexing loops rather than a generic slice iterator.
#
# Two methods are provided:
#   1. A simple 1-D overload that delegates to the underlying matrix.
#   2. A @generated N-D overload that loops over all indices except the
#      chosen dimension and applies the 1-D overload slice-by-slice.

# Index variable names used in the generated loop nest (supports up to rank 4).
const __VARS__ = (:i, :j, :k, :l)

"""
    mul!(y::AbstractVector, D::ChebDiff, x::AbstractVector) -> y

In-place matrix–vector product `y ← D * x`.

This is a thin wrapper around the underlying dense `Matrix` multiply and is
provided so that `ChebDiff` objects slot in everywhere an `AbstractMatrix` is
expected in `LinearAlgebra`.

# Arguments
- `y` : output vector (overwritten).
- `D` : Chebyshev differentiation matrix.
- `x` : input vector of CGL point values.

# Examples
```julia
julia> N  = 32;
julia> y  = chebpts(N);
julia> D  = chebdiff(N);
julia> dy = similar(y);
julia> mul!(dy, D, exp.(y))   # in-place first derivative of exp
```
"""
function LinearAlgebra.mul!(y::AbstractArray{S, 1},
                            D::ChebDiff{T},
                            x::AbstractArray{S, 1}) where {S, T}
    LinearAlgebra.mul!(y, D.mat, x)
end

"""
    mul!(y, D::ChebDiff, x, dim=Val(1)) -> y

In-place differentiation of an N-dimensional array `x` along dimension `dim`,
storing the result in `y`.

`D` must have been constructed for the size of `x` along `dim`.  The function
loops over every combination of indices in the *other* dimensions and applies
`D` to each resulting 1-D slice, so the cost is

    (size of x) / size(x, dim)  multiplications by D

with no heap allocations.

# Arguments
- `y`          : output array, same shape as `x` (overwritten).
- `D`          : `ChebDiff{T,N}` where `N == size(x, dim)`.
- `x`          : input array.
- `dim::Val`   : the dimension along which to differentiate (default `Val(1)`).

# Implementation note

The body of this function is `@generated`: Julia compiles a specialised version
for every concrete `(S, N, T, DIM)` combination at first call.  The generated
code is a simple nested `for` loop with `@inbounds @views` annotations,
equivalent to what you would write by hand for each specific dimensionality.

# Examples
```julia
julia> Ny, Nz = 32, 16;
julia> D  = chebdiff(Ny);
julia> # 3-D field: (z, y, t) layout — differentiate along dimension 2 (y)
julia> x  = rand(Nz, Ny, 8);
julia> dy = similar(x);
julia> mul!(dy, D, x, Val(2))
```
"""
@generated function LinearAlgebra.mul!(y::AbstractArray{S, N},
                                       D::ChebDiff{T},
                                       x::AbstractArray{S, N},
                                        ::Val{DIM}=Val(1)) where {S, N, T, DIM}
    # Build index expressions for one slice along DIM.
    # e.g. for N=3, DIM=2: x_slice = x[:, :, k]  →  x[i, :, k]
    x_slice = Expr(:ref, :x, ntuple(i -> i == DIM ? :(:) : __VARS__[i], N)...)
    y_slice = Expr(:ref, :y, ntuple(i -> i == DIM ? :(:) : __VARS__[i], N)...)

    # Symbols for each dimension size: N1, N2, …
    Ni = [Symbol(:N, d) for d in 1:N]

    # Build the loop nest over all dimensions except DIM (innermost first).
    loop_head = [Expr(:(=), __VARS__[d],
                      Expr(:call, :(:), 1, Symbol(:N, d)))
                 for d in N:-1:1 if d != DIM]
    loop_expr = Expr(:for, Expr(:block, loop_head...),
                     :(mul!($y_slice, D, $x_slice)))

    return quote
        # Unpack all dimension sizes in one assignment, e.g. N1, N2, N3 = size(y)
        $(Expr(:(=), Expr(:tuple, Ni...), :(size(y))))

        @inbounds @views begin
            $loop_expr
        end

        return y
    end
end
