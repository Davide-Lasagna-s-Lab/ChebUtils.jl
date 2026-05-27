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

"""
    mul!(y, D::ChebOp, x) -> y
    mul!(y, D::ChebOp, x, Val(DIM)) -> y

In-place Chebyshev spectral differentiation (or adjoint differentiation) of `x`,
storing the result in `y`.  `D` may be a [`ChebDiff`](@ref) (forward operator)
or an [`AdjointChebDiff`](@ref) (adjoint / weighted-adjoint operator).

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

julia> # Adjoint operator
julia> Dt = adjoint(D);
julia> w  = similar(f);
julia> mul!(w, Dt, f)                # w = Dᵀ f
```

See also [`chebdiff`](@ref), [`AdjointChebDiff`](@ref).
"""
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
