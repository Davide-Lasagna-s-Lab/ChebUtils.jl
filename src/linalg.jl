# Linear algebra extensions for ChebDiff (currently just lu!).

"""
    lu!(D::ChebDiff) -> LU

Compute the in-place LU factorisation of `D`, mutating its underlying storage.

The boundary rows of `D` are often replaced before factorisation to enforce
Dirichlet (or other) boundary conditions:

```julia
D[1, :] .= 0; D[1, 1] = 1   # pin top boundary
D[end, :] .= 0; D[end, end] = 1   # pin bottom boundary
F = lu!(D)
F \\ rhs   # solve the BVP
```

Returns a `LinearAlgebra.LU` factorisation object.
"""
LinearAlgebra.lu!(D::ChebDiff) = LinearAlgebra.lu!(D.mat)
