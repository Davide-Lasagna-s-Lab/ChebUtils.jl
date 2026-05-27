# API Reference

## Grid nodes and weights

```@docs
chebpts
chebws
```

## Differentiation matrices

```@docs
ChebDiff
chebdiff
chebddiff
```

## Adjoint

```@docs
AdjointChebDiff
LinearAlgebra.adjoint(::ChebDiff)
LinearAlgebra.adjoint(::AdjointChebDiff)
```

## Weighted adjoint

```@docs
LinearAlgebra.adjoint(::ChebDiff{T}, ::AbstractVector{T}) where {T}
```

## Multiplication

```@docs
LinearAlgebra.mul!(::AbstractArray{S,1}, ::ChebDiff, ::AbstractArray{S,1}) where {S}
LinearAlgebra.mul!(::AbstractArray{S,1}, ::AdjointChebDiff, ::AbstractArray{S,1}) where {S}
```

## Linear algebra

```@docs
LinearAlgebra.lu!(::ChebDiff)
```
