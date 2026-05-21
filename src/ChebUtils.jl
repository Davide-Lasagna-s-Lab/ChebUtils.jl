module ChebUtils

using LinearAlgebra

export chebpts, chebdiff, chebddiff, chebws, AdjointChebDiff

include("chebdiff.jl")
include("adjoint.jl")
include("constructors.jl")
include("clencurt.jl")
include("matmul.jl")
include("linalg.jl")

chebpts(N::Int, ::Type{T}=Float64) where {T} = T.(cos.((0:(N - 1))π/(N - 1)))

end
