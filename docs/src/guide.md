# Concepts & Conventions

## The Chebyshev–Gauss–Lobatto grid

ChebUtils works exclusively on the **Chebyshev–Gauss–Lobatto (CGL)** grid, also
called the Chebyshev extreme points or Chebyshev points of the second kind:

```math
x_j = \cos\!\left(\frac{\pi(j-1)}{N-1}\right), \quad j = 1, \ldots, N.
```

Key properties:

- `x₁ = +1`, `xₙ = −1` — the boundary nodes are always at the first and last
  indices.
- The ordering follows the Trefethen convention used in
  *Spectral Methods in MATLAB* (SIAM, 2000).
- The grid is **not** uniform; points cluster near both boundaries, which is
  essential for the spectral accuracy of differentiation.

`chebpts(N)` returns these nodes as a `Vector{Float64}` (or any other real type
via the second argument).

---

## Spectral differentiation

For a smooth function `f` sampled at the CGL nodes, the product `D * f` where
`D = chebdiff(N)` gives the derivative `f'` at each node with **spectral
(exponential) accuracy** — the error decays faster than any polynomial rate as
`N → ∞`.

### Construction: `chebdiff` and `chebddiff`

`chebdiff(N)` builds the `N×N` first-order matrix using the Fornberg / Trefethen
formula:

```math
D_{ij} = \frac{c_i}{c_j} \cdot \frac{(-1)^{i+j}}{x_i - x_j}, \quad i \neq j,
```

where `cⱼ = 2` for `j = 1` or `j = N` and `cⱼ = 1` otherwise.  Diagonal
entries are set by the **negative-sum trick** (`Dᵢᵢ = −∑_{j≠i} Dᵢⱼ`), ensuring
exact differentiation of constants.

`chebddiff(N)` computes the second-order matrix using an analogous closed-form
formula applied directly to `D`.  This is more numerically stable than squaring
`D * D`.

### The `ChebDiff` type

Both constructors return a `ChebDiff{T, N} <: AbstractMatrix{T}`, which wraps
the dense `N×N` matrix and implements the full `AbstractMatrix` interface
(indexing, broadcasting, LU factorisation).  Passing a `ChebDiff` where an
`AbstractMatrix` is expected works without any conversion.

---

## Multi-dimensional differentiation

The core `mul!` method differentiates along a single dimension of an arbitrary
N-dimensional array:

```julia
mul!(y, D, x, Val(DIM))
```

This applies `D` to each 1-D fibre of `x` along dimension `DIM` and stores the
result in `y`.  The loop nest is built by a `@generated` function at compile
time — there are no views, no closures, and no runtime dispatch overhead.

```julia
# Differentiate a (Nx, Ny, Nz) array along the wall-normal axis 2
Ny = 64
D  = chebdiff(Ny)
F  = randn(32, Ny, 16)
dF = similar(F)
mul!(dF, D, F, Val(2))   # dF[:, i, :] ≈ ∂F/∂y at node i
```

The 1-D form `mul!(y, D, x)` (without `Val`) differentiates vectors and is
equivalent to `y = D.mat * x`.

---

## Clenshaw-Curtis quadrature

`chebws(N)` returns the `N` Clenshaw-Curtis quadrature weights `w` satisfying

```math
\int_{-1}^{1} f(x)\, dx \approx \sum_{j=1}^{N} w_j\, f(x_j),
```

with spectral convergence for smooth `f` and exact integration for polynomials
of degree ≤ `N−1`.  The algorithm is the DCT-based method of Trefethen.

The weights define the **weighted inner product**

```math
\langle u, v \rangle_W = \sum_{j=1}^{N} w_j\, u(x_j)\, v(x_j),
```

which is used, for example, to define the adjoint of the differentiation
operator.

---

## Adjoints

### Euclidean adjoint

`adjoint(D)` returns an `AdjointChebDiff` whose stored matrix is `transpose(D)`.
For the standard Euclidean inner product this satisfies `⟨v, D u⟩ = ⟨Dᵀ v, u⟩`.

### Weighted adjoint

`adjoint(D, w)` returns the **W-weighted adjoint** `D† = W⁻¹ D' W` where
`W = Diagonal(w)`.  Entry-wise: `(D†)ᵢⱼ = Dⱼᵢ · wⱼ / wᵢ`.  This satisfies

```math
\langle v, D u \rangle_W = \langle D^\dagger v, u \rangle_W.
```

Pass `w = chebws(N)` to obtain the spectrally consistent weighted adjoint.

### Precomputation

Both forms precompute the full adjoint matrix at construction time.  All
subsequent `mul!` calls on `AdjointChebDiff` are allocation-free dense
matrix-vector multiplies — identical cost to the forward operator.

---

## Boundary-value problems

The standard workflow for a second-order BVP with Dirichlet boundary conditions:

```julia
N  = 64
DD = chebddiff(N)

# Replace boundary rows to enforce u(+1) = u(-1) = 0
DD[1,   :] .= 0;  DD[1,   1]   = 1
DD[end, :] .= 0;  DD[end, end] = 1

# LU factorise (mutates DD)
F   = lu!(DD)

# Right-hand side: Poisson equation −u'' = f, u(±1) = 0
rhs = -sin.(chebpts(N))
rhs[1] = rhs[end] = 0

u = F \ rhs
```

`lu!` mutates the underlying storage of `ChebDiff` and returns a
`LinearAlgebra.LU` factorisation object.
