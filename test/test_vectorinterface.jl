import VectorInterface as VI
using ITensorBase: dimnames, named, unnamed
using Test: @test, @testset

# These name-aware methods are what let an NamedTensor be used directly as a vector in
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
    @test VI.scalartype([a, b]) === elt
    @test VI.scalartype([a, randn(ComplexF64, i, j)]) === ComplexF64

    # zerovector / zerovector! / zerovector!!
    z = VI.zerovector(a, ComplexF64)
    @test VI.scalartype(z) === ComplexF64
    @test iszero(unnamed(z))
    @test dimnames(z) == dimnames(a)
    z = VI.zerovector!(copy(a))
    @test VI.scalartype(z) === elt
    @test iszero(unnamed(z))
    @test VI.zerovector!!(a) ≈ VI.zerovector(a)

    # scale / scale! / scale!!
    @test unnamed(VI.scale(a, 2)) ≈ 2 * ua
    @test unnamed(VI.scale!(copy(a), 2)) ≈ 2 * ua
    @test unnamed(VI.scale!(similar(a), a, 2)) ≈ 2 * ua
    @test unnamed(VI.scale!!(copy(a), 2)) ≈ 2 * ua
    @test unnamed(VI.scale!!(similar(a), a, 2)) ≈ 2 * ua
    # `!!` allocates a new array when the scalar promotes beyond the element type.
    s = VI.scale!!(copy(a), 2im)
    @test VI.scalartype(s) === complex(elt)
    @test unnamed(s) ≈ 2im * ua

    # add / add! / add!!
    @test unnamed(VI.add(b, a), dimnames(a)) ≈ ub + ua
    @test unnamed(VI.add(b, a, 2, 3), dimnames(a)) ≈ 3 * ub + 2 * ua
    @test unnamed(VI.add!(copy(b), a, 2, 3), dimnames(a)) ≈ 3 * ub + 2 * ua
    @test unnamed(VI.add!!(copy(b), a, 2, 3), dimnames(a)) ≈ 3 * ub + 2 * ua
    r = VI.add!!(copy(b), a, 2im, 3)
    @test VI.scalartype(r) === complex(elt)
    @test unnamed(r, dimnames(a)) ≈ 3 * ub + 2im * ua

    @test VI.inner(a, b) ≈ VI.inner(ua, ub)
end
