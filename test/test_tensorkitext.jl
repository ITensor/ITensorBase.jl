using ITensorBase: ITensorBase, Index, dimnames, name, unnamed
using LinearAlgebra: norm
using MatrixAlgebraKit: qr_compact, svd_compact
using StableRNGs: StableRNG
using TensorKit: TensorKit, @tensor, AbstractTensorMap, SU2Irrep, U1Irrep, Vect, dim, dual,
    scalar, space, ⊗
using Test: @test, @test_throws, @testset

# A native TensorKit space flows into `Index`, so an `ITensor` wraps a `TensorMap` directly.
# Cover an abelian (U₁) and a non-abelian (SU₂) symmetry; the non-abelian case is the point
# — no block-sparse abelian backend can represent it.
@testset "TensorKitExt (eltype = $elt)" for elt in (Float64, ComplexF64)
    rng = StableRNG(1234)
    @testset "$label" for (label, Vi, Vj, Vk) in (
            (
                "U₁",
                Vect[U1Irrep](0 => 2, 1 => 3),
                Vect[U1Irrep](0 => 1, 1 => 2),
                Vect[U1Irrep](-1 => 1, 0 => 2),
            ),
            (
                "SU₂",
                Vect[SU2Irrep](0 => 2, 1 // 2 => 1),
                Vect[SU2Irrep](1 // 2 => 1, 1 => 1),
                Vect[SU2Irrep](0 => 1, 1 // 2 => 2),
            ),
        )
        i, j, k = Index(Vi), Index(Vj), Index(Vk)

        # `Index` stores the native space directly.
        @test unnamed(i) === Vi

        # `conj(index)` round-trips to an `Index` carrying the dual space, same name.
        @test conj(i) isa Index
        @test unnamed(conj(i)) == dual(Vi)
        @test name(conj(i)) == name(i)

        # Cold-start construction wraps a `TensorMap`; size/eltype report dense values.
        a = randn(rng, elt, i, j)
        @test unnamed(a) isa AbstractTensorMap
        @test size(a) == (dim(Vi), dim(Vj))
        @test eltype(a) == elt
        @test norm(unnamed(zeros(elt, i, j))) == 0

        # Contraction over the shared (dualized) leg matches a direct TensorKit reference.
        b = randn(rng, elt, conj(j), k)
        c = a * b
        @test Set(dimnames(c)) == Set(name.((i, k)))
        ta, tb, gc = unnamed(a), unnamed(b), unnamed(c)
        @tensor ref[vi; vk] := ta[vi, vj] * tb[vj, vk]
        @test space(ref) == space(gc)
        @test ref ≈ gc

        # Linear-combination broadcast lowers to `bipermutedimsopadd!`; element-wise errors.
        b2 = randn(rng, elt, i, j)
        @test unnamed(a + b2) ≈ unnamed(a) + unnamed(b2)
        @test unnamed(2 * a) ≈ 2 * unnamed(a)
        @test unnamed(a .- 3 .* b2) ≈ unnamed(a) - 3 * unnamed(b2)
        @test_throws ErrorException sin.(a)

        # Factorizations reconstruct the tensor (lowered through matricize / MatrixAlgebraKit).
        # Checked by full contraction to a scalar, which is bipartition-independent.
        a3 = randn(rng, elt, i, j, k)
        w = randn(rng, elt, conj(i), conj(j), conj(k))
        sca(x) = scalar(unnamed(x))
        u, s, v = svd_compact(a3, (i,), (j, k))
        @test sca((u * s * v) * w) ≈ sca(a3 * w)
        q, r = qr_compact(a3, (i,), (j, k))
        @test sca((q * r) * w) ≈ sca(a3 * w)
    end
end
