using ITensorBase: ITensorBase, Index, aligndims, dimnames, name, prime, unnamed
using LinearAlgebra: norm
using MatrixAlgebraKit: qr_compact, svd_compact
using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, project, unchecked_project
using TensorKit: TensorKit, @tensor, AbstractTensorMap, SU2Irrep, U1Irrep, Vect, dim, dual,
    scalar, space, ←, ⊗
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

        # `isdual`/`dual` forward to the underlying space.
        @test TensorAlgebra.isdual(i) == false
        @test TensorAlgebra.isdual(conj(i)) == true
        @test unnamed(TensorAlgebra.dual(i)) == dual(Vi)
        @test name(TensorAlgebra.dual(i)) == name(i)

        # Equality is dual-insensitive: an index equals its dual (same name, same ungraded
        # extent), hashes match, and a fresh index of the same space is a distinct leg. This is
        # what lets `Base` set-ops / `Dict` / `Set` treat an index and its dual as one leg on a
        # symmetric backend.
        @test conj(i) == i
        @test isequal(conj(i), i)
        @test hash(conj(i)) == hash(i)
        @test Index(Vi) != i

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

        # `Array` densifies a `TensorMap`-backed ITensor to a plain dense array (a fresh copy).
        @test Array(a) isa Array{elt}
        @test Array(a) == convert(Array, unnamed(a))

        # `trivialrange` mints a fresh trivial axis over the native space (used e.g. by the
        # boundary-MPS setup), routing through the `namedunitrange(::ElementarySpace, name)` overload.
        t1 = TensorAlgebra.trivialrange(i)
        @test t1 isa Index
        @test length(t1) == 1
        tn = TensorAlgebra.trivialrange(i, 3)
        @test tn isa Index
        @test length(tn) == 3
        @test name(tn) != name(i)

        # Map-shaped construction forwards the codomain/domain split to the `TensorMap`:
        # `randn((i,), (j,))` stores a `Vi ← Vj` map rather than flattening all-codomain.
        # Following the `similar_map` convention the domain appears dualized in the outward
        # view, the same as a flat graded array would have axes `(Vi, dual(Vj))`.
        m = randn(rng, elt, (i,), (j,))
        @test unnamed(m) isa AbstractTensorMap
        @test space(unnamed(m)) == (Vi ← Vj)
        @test space(unnamed(m), 1) == Vi
        @test space(unnamed(m), 2) == dual(Vj)
        @test unnamed(rand(rng, elt, (i,), (j,))) isa AbstractTensorMap
        @test norm(unnamed(zeros(elt, (i,), (j,)))) == 0
        # The friendly forms agree with the underlying `TensorAlgebra` map hooks.
        @test space(unnamed(TensorAlgebra.randn_map(elt, (i, j), (k,)))) ==
            space(unnamed(randn(elt, (i, j), (k,))))
        # An empty codomain builds an all-domain `TensorMap`, the mirror of an empty domain. The
        # space type is read from the domain, since the empty codomain carries none. An all-empty
        # split has no map meaning and errors rather than recursing.
        cd = randn(rng, elt, (), (j,))
        @test unnamed(cd) isa AbstractTensorMap
        @test space(unnamed(cd)) == (one(Vj) ← Vj)
        @test dimnames(cd) == [name(j)]
        @test space(unnamed(zeros(elt, (), (j,)))) == (one(Vj) ← Vj)
        @test_throws MethodError randn(rng, elt, (), ())

        # `aligndims` reorders a `TensorMap`-backed tensor. The flat form gives an all-codomain
        # result and the map form re-expresses the requested codomain/domain split, both
        # carrying each index with its arrow to the new position.
        mf = aligndims(m, (j, i))
        @test dimnames(mf) == [name(j), name(i)]
        @test space(unnamed(mf), 1) == dual(Vj)
        @test space(unnamed(mf), 2) == Vi
        md = aligndims(m, (j,), (i,))
        @test dimnames(md) == [name(j), name(i)]
        @test space(unnamed(md)) == (dual(Vj) ← dual(Vi))
        @test space(unnamed(md), 1) == dual(Vj)
        @test space(unnamed(md), 2) == Vi
        # An empty codomain moves both indices into the domain, preserving the outward axes.
        me = aligndims(m, (), (i, j))
        @test dimnames(me) == [name(i), name(j)]
        @test space(unnamed(me)) == (one(Vi) ← (dual(Vi) ⊗ Vj))
        @test space(unnamed(me), 1) == Vi
        @test space(unnamed(me), 2) == dual(Vj)
    end

    # `project` builds a `TensorMap`-backed operator/state from a dense basis matrix: the index
    # spaces select the backend, so the same call that yields an `Array` on dense indices yields a
    # `TensorMap` here, keeping only the symmetry-allowed blocks.
    @testset "project" begin
        W = Vect[U1Irrep](0 => 1, 1 => 1)
        w = Index(W)
        Sz = elt[0.5 0; 0 -0.5]
        Sx = elt[0 0.5; 0.5 0]

        top = project(Sz, (prime(w),), (w,))
        @test unnamed(top) isa AbstractTensorMap
        @test space(unnamed(top)) == (W ← W)
        @test Set(dimnames(top)) == Set(name.((prime(w), w)))

        # a charge-breaking operator is projected to zero by `unchecked_project`; the checked
        # `project` rejects the discard
        @test norm(unnamed(unchecked_project(Sx, (prime(w),), (w,)))) == 0
        @test_throws InexactError project(Sx, (prime(w),), (w,); atol = 0, rtol = 0)

        # the two-argument form builds an all-codomain state; only the trivial-charge component
        # of the dense vector survives
        pv = project(elt[1, 0], (w,))
        @test unnamed(pv) isa AbstractTensorMap
        @test norm(unnamed(pv)) ≈ 1
        @test norm(unnamed(unchecked_project(elt[0, 1], (w,)))) == 0

        # the empty-codomain form builds an all-domain `TensorMap` (the mirror case)
        cobra = project(elt[1, 0], (), (w,))
        @test unnamed(cobra) isa AbstractTensorMap
        @test space(unnamed(cobra)) == (one(W) ← W)
        @test Set(dimnames(cobra)) == Set((name(w),))
    end
end
