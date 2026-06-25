import VectorInterface as VI
using ITensorBase: dimnames, named, unnamed
using Test: @test, @testset

# These name-aware methods are what let an ITensor be used directly as a vector in
# iterative solvers such as `KrylovKit.eigsolve`, which drive their Krylov vectors
# through `VectorInterface`.
@testset "VectorInterface (eltype=$(elt))" for elt in
    (Float32, Float64, Complex{Float32})
    i, j = named.(2, (:i, :j))
    a = randn(elt, i, j)
    b = randn(elt, j, i)
    ua = unnamed(a)
    ub = unnamed(b, dimnames(a))

    @test VI.scalartype(a) === elt

    z = VI.zerovector(a, ComplexF64)
    @test VI.scalartype(z) === ComplexF64
    @test iszero(unnamed(z))
    @test dimnames(z) == dimnames(a)

    @test unnamed(VI.scale(a, 2)) ≈ 2 * ua
    @test unnamed(VI.scale!!(a, 2)) ≈ 2 * ua
    @test unnamed(VI.scale!!(copy(a), a, 2)) ≈ 2 * ua

    @test unnamed(VI.add!!(copy(b), a, 2, 3), dimnames(a)) ≈ 2 * ua + 3 * ub

    @test VI.inner(a, b) ≈ VI.inner(ua, ub)
end
