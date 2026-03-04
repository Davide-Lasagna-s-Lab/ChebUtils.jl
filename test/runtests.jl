using ChebUtils
using LinearAlgebra
using Random
using Test

@testset "Constructors              " begin
    # generate differentiation matrices and add complex part
    D = chebdiff(4)
    E = 6im*I + D

    @test D isa ChebUtils.ChebDiff{Float64, 4}
    @test E isa Matrix{ComplexF64}
end

@testset "Points                    " begin
    # generate random integer in range
    rint = rand(3:64)
    rand_ind = rand(2:(rint - 1))

    # evaluate points
    points = chebpts(rint)

    @test size(points) == (rint,)
    @test points[1] == 1.0
    @test points[end] == -1.0
    @test points[rand_ind] ≈ Float64(cos(π*(rand_ind - 1)/(rint - 1)))
end

@testset "Differentiation matrices  " begin
    # generate random integer in range
    rint = rand(2:64)

    # generate first order differential matrices
    diffmat2 = chebdiff(2)
    diffmat3 = chebdiff(3)
    diffmat4 = chebdiff(4)
    diffmat_randsize = chebdiff(rint)

    # generate the second order differential matrix
    double_diffmat_randsize = chebddiff(rint)

    # size check for random size matrices
    @test size(diffmat_randsize) == (rint, rint)
    @test size(double_diffmat_randsize) == (rint, rint)

    # first order matrices correct
    @test diffmat2 ≈ [0.5 -0.5; 0.5 -0.5]
    @test diffmat3 ≈ [1.5 -2.0 0.5; 0.5 0.0 -0.5; -0.5 2.0 -1.5]
    @test diffmat4 ≈ [19/6 -4.0 4/3 -0.5; 1.0 -1/3 -1.0 1/3; -1/3 1.0 1/3 -1; 0.5 -4/3 4.0 -19/6]

    # second order matrices correct (based off first order)
    @test double_diffmat_randsize.mat ≈ diffmat_randsize.mat*diffmat_randsize.mat
end

@testset "Quadrature weights        " begin
    # non-polynomial function
    Ny1 = 16
    y1 = chebpts(Ny1)
    ws1 = chebws(Ny1)
    I1 = exp(1) - exp(-1)
    @test abs(sum(ws1.*exp.(y1)) - I1) < 1e-8

    # polynomial function
    Ny2 = 3
    y2 = chebpts(Ny2)
    ws2 = chebws(Ny2)
    @test abs(sum(ws2.*(x->x^3).(y2))) < 1e-15
end

@testset "Matmul of vector          " begin
    # initialise differentiation matrices
    N = 32
    y = chebpts(N)
    D = chebdiff(N); DD = chebddiff(N)

    # generate field to be differentiatied
    fs_fun(y) = exp(1.1*y)
    fs = fs_fun.(y)

    # generate exact derivative fields
    dfs_fun(y) = 1.1*fs_fun(y)
    ddfs_fun(y) = (1.1^2)*fs_fun(y)
    dfs_EX = dfs_fun.(y)
    ddfs_EX = ddfs_fun.(y)

    # compute derivative using matrix
    dfs_FD = zero(fs)
    ddfs_FD = zero(fs)
    mul!(dfs_FD, D, fs)
    mul!(ddfs_FD, DD, fs)

    @test dfs_FD ≈ dfs_EX
    @test ddfs_FD ≈ ddfs_EX
end

@testset "Matmul of cube            " begin
    # initialise differentiation matrices
    Ny = 32; Nz = 32; Nt = 32
    grid = (reshape((0:(Nz - 1))/Nz*2π, :, 1, 1), reshape(chebpts(Ny), 1, :, 1), reshape((0:(Nt - 1))/Nt*2π, 1, 1, :))
    D = chebdiff(Ny); DD = chebddiff(Ny)

    # generate field to be differentiatied
    fs_fun(z, y, t) = exp(1.1*y)*exp(cos(z))*atan(sin(t))
    fs = fs_fun.(grid...)

    # generate exact derivative fields
    dfs_fun(z, y, t) = 1.1*fs_fun(z, y, t)
    ddfs_fun(z, y, t) = (1.1^2)*fs_fun(z, y, t)
    dfs_EX = dfs_fun.(grid...)
    ddfs_EX = ddfs_fun.(grid...)

    # compute derivative using matrix
    dfs_FD = zero(fs)
    ddfs_FD = zero(fs)
    mul!(dfs_FD, D, fs, Val(2))
    mul!(ddfs_FD, DD, fs, Val(2))

    @test dfs_FD ≈ dfs_EX
    @test ddfs_FD ≈ ddfs_EX
end

@testset "Matmul of hypercube       " begin
    # initialise differentiation matrices
    Ny = 32; Nx=8; Nz = 32; Nt = 32
    grid = (reshape((0:(Nx - 1))/Nx*2π, :, 1, 1, 1),
            reshape((0:(Nz - 1))/Nz*2π, 1, :, 1, 1),
            reshape((0:(Nt - 1))/Nt*2π, 1, 1, :, 1),
            reshape(chebpts(Ny),        1, 1, 1, :))
    D = chebdiff(Ny); DD = chebddiff(Ny)

    # generate field to be differentiatied
    fs_fun(x, z, t, y) = exp(1.1*y)*cos(x)*exp(cos(z))*atan(sin(t))
    fs = fs_fun.(grid...)

    # generate exact derivative fields
    dfs_fun(x, z, t, y) = 1.1*fs_fun(x, z, t, y)
    ddfs_fun(x, z, t, y) = (1.1^2)*fs_fun(x, z, t, y)
    dfs_EX = dfs_fun.(grid...)
    ddfs_EX = ddfs_fun.(grid...)

    # compute derivative using matrix
    dfs_FD = zero(fs)
    ddfs_FD = zero(fs)
    mul!(dfs_FD, D, fs, Val(4))
    mul!(ddfs_FD, DD, fs, Val(4))

    @test dfs_FD ≈ dfs_EX
    @test ddfs_FD ≈ ddfs_EX
end

@testset "LU decomposition          " begin
    # initialise differentiation matrices
    N = 16
    y = chebpts(N)
    D = chebdiff(N); DD = chebddiff(N)

    # make them invertible
    D[1, :] .= 0; D[1, 1] = 1
    DD[1, :] .= 0; DD[1, 1] = 1
    DD[end, :] .= 0; DD[end, end] = 1

    # make copy to be compared later
    Dbase = copy(parent(D)); DDbase = copy(parent(DD))

    # compute LU decomposition
    D_LU = lu!(D)
    DD_LU = lu!(DD)

    # check types
    @test D_LU isa LinearAlgebra.LU
    @test DD_LU isa LinearAlgebra.LU

    # compare reconstructed values
    @test D_LU.L * D_LU.U ≈ Dbase[D_LU.p, :]
    @test DD_LU.L * DD_LU.U ≈ DDbase[DD_LU.p, :]
end
