import LinearAlgebra as LA
using GradedArrays: SU2, Z2, fSU2, fZ2, gradedrange
using ITensorBase: ITensorBase, Named, NamedTensor, unname, unnamed
using StableRNGs: StableRNG
using TensorAlgebra: bipermutedims
using TensorKit: TensorKit
using Test: @test, @testset
using VectorInterface: VectorInterface as VI

@testset "LinearAlgebra (eltype=$(elt))" for elt in
    (Float32, Float64, Complex{Float32})
    i, j = Named.(2, (:i, :j))
    a = randn(elt, i, j)
    b = randn(elt, j, i)
    @test LA.norm(a) ≈ LA.norm(unnamed(a))
    @test unnamed(LA.normalize(a)) ≈ LA.normalize(unnamed(a))
    @test unnamed(LA.normalize!(copy(a))) ≈ LA.normalize(unnamed(a))
    @test unnamed(LA.rmul!(copy(a), 2)) ≈ 2 * unnamed(a)
    @test unnamed(LA.lmul!(2, copy(a))) ≈ 2 * unnamed(a)
    @test unnamed(LA.rdiv!(copy(a), 2)) ≈ unnamed(a) / 2
    @test unnamed(LA.ldiv!(2, copy(a))) ≈ 2 \ unnamed(a)
    @test LA.dot(a, b) ≈ LA.dot(unnamed(a), unname(b, ITensorBase.names(a)))
end

@testset "Graded inner product ($G, $T, split=$n)" for (G, g) in (
            ("Z2", gradedrange([Z2(0) => 2, Z2(1) => 1])),
            ("fZ2", gradedrange([fZ2(false) => 2, fZ2(true) => 1])),
            ("SU2", gradedrange([SU2(0) => 2, SU2(1 // 2) => 1, SU2(1) => 1])),
            ("fSU2", gradedrange([fSU2(0) => 2, fSU2(1 // 2) => 1, fSU2(1) => 1])),
        ),
        T in (Float64, ComplexF64),
        n in 0:3

    rng = StableRNG(123)
    cod = ntuple(_ -> g, n)
    dom = ntuple(_ -> g, 3 - n)
    x = randn(rng, T, cod, dom)
    y = randn(rng, T, cod, dom)
    a = NamedTensor(x, (:i, :j, :k))
    b = NamedTensor(y, (:i, :j, :k))
    expected = LA.dot(TensorKit.TensorMap(x), TensorKit.TensorMap(y))
    @test LA.dot(a, a) ≈ LA.norm(a)^2
    @test LA.dot(a, b) ≈ expected
    @test VI.inner(a, b) ≈ expected
    @test LA.dot(a, b) ≈ conj(LA.dot(b, a))
    for m in 0:3
        repartitioned =
            NamedTensor(bipermutedims(y, Tuple(1:m), Tuple((m + 1):3)), (:i, :j, :k))
        @test repartitioned ≈ b
        @test LA.dot(a, repartitioned) ≈ expected
        perm = (3, 1, 2)
        reordered = NamedTensor(bipermutedims(y, perm[1:m], perm[(m + 1):3]), (:k, :i, :j))
        @test reordered ≈ b
        @test LA.dot(a, reordered) ≈ expected
        @test LA.dot(reordered, a) ≈ conj(expected)
    end
end
