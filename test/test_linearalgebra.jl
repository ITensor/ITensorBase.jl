import LinearAlgebra as LA
using ITensorBase: dimnames, named, unnamed
using Test: @test, @testset

@testset "LinearAlgebra (eltype=$(elt))" for elt in
    (Float32, Float64, Complex{Float32})
    i, j = named.(2, (:i, :j))
    a = randn(elt, i, j)
    b = randn(elt, j, i)
    @test LA.norm(a) ≈ LA.norm(unnamed(a))
    @test unnamed(LA.normalize(a)) ≈ LA.normalize(unnamed(a))
    @test unnamed(LA.normalize!(copy(a))) ≈ LA.normalize(unnamed(a))
    @test unnamed(LA.rmul!(copy(a), 2)) ≈ 2 * unnamed(a)
    @test unnamed(LA.lmul!(2, copy(a))) ≈ 2 * unnamed(a)
    @test unnamed(LA.rdiv!(copy(a), 2)) ≈ unnamed(a) / 2
    @test unnamed(LA.ldiv!(2, copy(a))) ≈ 2 \ unnamed(a)
    @test LA.dot(a, b) ≈ LA.dot(unnamed(a), unnamed(b, dimnames(a)))
end
