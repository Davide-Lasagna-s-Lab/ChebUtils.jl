# LinearAlgebra extensions for ChebDiff.
#
# ChebDiff wraps a plain Matrix, so most LinearAlgebra operations work
# automatically via the AbstractMatrix interface.  The one exception is
# lu!, which needs to forward to the underlying storage because Julia's
# default pivoting code does not know how to mutate ChebDiff in-place.

"""
    lu!(D::ChebDiff) -> LinearAlgebra.LU

Compute the LU factorisation of `D` **in-place**, overwriting the underlying
matrix storage.

This is the standard `LinearAlgebra.lu!` method forwarded to `D.mat`.  It is
useful for solving linear systems of the form `D * x = b` (e.g. boundary-value
problems with row-replaced boundary conditions) without allocating a copy of
the matrix.

# Notes

Chebyshev differentiation matrices are singular as constructed because constant
functions are in their null space.  Before calling `lu!`, replace the rows
corresponding to the boundary conditions with the constraint equations.  For
example, to impose homogeneous Dirichlet conditions at both ends:

```julia
D[1,   :] .= 0;  D[1,   1]   = 1
D[end, :] .= 0;  D[end, end] = 1
```

# Examples
```julia
julia> N  = 16;
julia> DD = chebddiff(N);
julia> # Impose Dirichlet BCs
julia> DD[1, :] .= 0; DD[1, 1] = 1
julia> DD[N, :] .= 0; DD[N, N] = 1
julia> F  = lu!(DD)          # factorises DD.mat in-place
julia> b  = ones(N); b[1] = b[N] = 0
julia> x  = F \\ b            # solve the system
```

See also [`chebddiff`](@ref).
"""
LinearAlgebra.lu!(D::ChebDiff) = LinearAlgebra.lu!(D.mat)
