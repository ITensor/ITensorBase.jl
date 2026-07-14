using ITensorBase: ITensorBase, Index, dimnames, id, inds, name, namedoneto, operator,
    prime, replacedimnames, uniquename, unname, unnamed
using LinearAlgebra: LinearAlgebra, norm, tr
using MatrixAlgebraKit: left_null, left_orth, left_polar, lq_compact, lq_full, qr_compact,
    qr_full, right_null, right_orth, right_polar, svd_compact, svd_trunc, svd_vals
using StableRNGs: StableRNG
using TensorAlgebra.MatrixAlgebra: gram_eigh_full, gram_eigh_full_with_pinv
using TensorAlgebra: TensorAlgebra, contract, directsum, matricize, project, trivialrange,
    unchecked_project, unmatricize
using Test: @test, @test_broken, @testset

@testset "TensorAlgebra (eltype=$(elt))" for elt in
    (
        Float32,
        Float64,
        Complex{Float32},
        Complex{Float64},
    )
    @testset "contract" begin
        i = namedoneto(2, "i")
        j = namedoneto(2, "j")
        k = namedoneto(2, "k")
        na1 = randn(elt, i, j)
        na2 = randn(elt, j, k)
        na_dest = na1 * na2
        @test eltype(na_dest) ≡ elt
        @test unname(na_dest, (i, k)) ≈ unnamed(na1) * unnamed(na2)
    end
    @testset "matricize" begin
        i, j, k, l = namedoneto.((2, 3, 4, 5), ("i", "j", "k", "l"))
        na = randn(elt, i, j, k, l)
        na_fused = matricize(na, (k, i) => "a", (j, l) => "b")
        # Fuse all dimensions.
        @test unname(na_fused, ("a", "b")) ≈ reshape(
            unname(na, (k, i, j, l)),
            (
                length(k) * length(i),
                length(j) * length(l),
            )
        )
        # Positional form auto-generates the two fused dimension names, matching the
        # pair form's data.
        na_pos = matricize(na, (k, i), (j, l))
        @test ndims(na_pos) == 2
        @test Array(na_pos) == Array(na_fused)
        # Groups may be any iterable of dimensions, not only tuples (no `Tuple` wrapping
        # needed), in both the pair and positional forms.
        @test Array(matricize(na, [k, i] => "a", [j, l] => "b")) == Array(na_fused)
        @test Array(matricize(na, [k, i], [j, l])) == Array(na_pos)
    end
    @testset "unmatricize" begin
        a, b = namedoneto.((6, 20), ("a", "b"))
        i, j, k, l = namedoneto.((2, 3, 4, 5), ("i", "j", "k", "l"))
        na = randn(elt, a, b)
        # Split all dimensions.
        na_split = unmatricize(na, "a" => (k, i), "b" => (j, l))
        @test unname(na_split, ("k", "i", "j", "l")) ≈
            reshape(
            unname(na, ("a", "b")),
            (unnamed(k), unnamed(i), unnamed(j), unnamed(l))
        )
    end
    @testset "directsum" begin
        i = namedoneto(2, "i")            # shared index, carried through unchanged
        j1, j2 = namedoneto.((2, 3), ("j1", "j2"))
        k1, k2 = namedoneto.((2, 3), ("k1", "k2"))
        a = randn(elt, i, j1, k1)
        b = randn(elt, i, j2, k2)
        ref = cat(unname(a, (i, j1, k1)), unname(b, (i, j2, k2)); dims = (2, 3))
        # Fresh output indices: the shared index is kept and the summed indices trail it,
        # and the minted indices are returned as a second output.
        s, summed = directsum(a => (j1, k1), b => (j2, k2))
        @test eltype(s) ≡ elt
        @test i in inds(s)
        @test issetequal(inds(s), (i, summed...))
        @test sort(length.(summed)) == [5, 5]
        @test unname(s, (i, summed...)) == ref
        # Explicit output indices name the summed dimensions.
        o1, o2 = namedoneto.((5, 5), ("o1", "o2"))
        s2 = directsum((o1, o2), a => (j1, k1), b => (j2, k2))
        @test issetequal(inds(s2), (i, o1, o2))
        @test unname(s2, (i, o1, o2)) == ref
        # A single summed dimension.
        u1, u2 = namedoneto.((2, 3), ("u1", "u2"))
        c = randn(elt, i, u1)
        d = randn(elt, i, u2)
        sc, (su,) = directsum(c => (u1,), d => (u2,))
        @test length(su) == 5
        @test unname(sc, (i, su)) == cat(unname(c, (i, u1)), unname(d, (i, u2)); dims = 2)
    end
    @testset "Matrix functions" begin
        for f in ITensorBase.MATRIX_FUNCTIONS
            f == :cbrt && elt <: Complex && continue
            f == :cbrt && VERSION < v"1.11-" && continue
            @eval begin
                i, j, k, l = namedoneto.((2, 2, 2, 2), ("i", "j", "k", "l"))
                rng = StableRNG(123)
                a = randn(rng, $elt, (i, j, k, l))
                fa = $f(a, (j, l), (k, i))
                m = unname(matricize(a, (j, l) => "a", (k, i) => "b"), ("a", "b"))
                fm = unname(matricize(fa, (j, l) => "a", (k, i) => "b"), ("a", "b"))
                @test fm ≈ $f(m)
            end
        end
    end
    @testset "qr/lq" begin
        dims = (2, 2, 2, 2)
        i, j, k, l = namedoneto.(dims, ("i", "j", "k", "l"))

        a = randn(elt, i, j)
        # TODO: Should this be allowed?
        # TODO: Add support for specifying new name.
        for f in (
                left_orth, left_polar, lq_compact, lq_full, qr_compact, qr_full,
                right_orth, right_polar,
            )
            x, y = f(a, (i,))
            @test x * y ≈ a
        end

        a = randn(elt, i, j, k, l)
        # TODO: Add support for specifying new name.
        for f in (
                left_orth, left_polar, lq_compact, lq_full, qr_compact, qr_full,
                right_orth, right_polar,
            )
            x, y = f(a, (i, k), (j, l))
            @test x * y ≈ a
        end
    end
    @testset "svd" begin
        dims = (2, 2, 2, 2)
        i, j, k, l = namedoneto.(dims, ("i", "j", "k", "l"))

        a = randn(elt, i, j)
        # TODO: Should this be allowed?
        # TODO: Add support for specifying new name.
        u, s, v = svd_compact(a, (i,))
        @test u * s * v ≈ a

        a = randn(elt, i, j, k, l)
        # TODO: Add support for specifying new name.
        u, s, v = svd_compact(a, (i, k), (j, l))
        @test u * s * v ≈ a

        # Test truncation. `svd_trunc` returns a fourth output `ϵ`, the truncation error
        # (2-norm of the discarded singular values), matching MatrixAlgebraKit.
        a = randn(elt, i, j, k, l)
        res = svd_trunc(a, (i, k), (j, l); trunc = (; maxrank = 2))
        @test length(res) == 4
        u, s, v, ϵ = res
        @test u * s * v ≉ a
        @test size(s) == (2, 2)
        @test ϵ isa Real
        @test ϵ ≥ 0
        # `ϵ` equals the 2-norm of the discarded singular values.
        vals = svd_vals(a, (i, k), (j, l))
        @test ϵ ≈ norm(sort(vals; rev = true)[3:end])
    end
    @testset "left_null/right_null" begin
        dims = (2, 2, 2, 2)
        i, j, k, l = namedoneto.(dims, ("i", "j", "k", "l"))

        a = randn(elt, i, j, k, l)
        # TODO: Add support for specifying new name.
        for n in (left_null(a, (i, k), (j, l)), left_null(a, (i, k)))
            @test (i, k) ⊆ inds(n)
            @test norm(n * a) ≈ 0
        end
        for n in (right_null(a, (i, k), (j, l)), right_null(a, (i, k)))
            @test (j, l) ⊆ inds(n)
            @test norm(n * a) ≈ 0
        end
    end
    @testset "gram_eigh_full" begin
        # Build a Hermitian PSD a ≈ conj(b) * b over an aux dim, with codomain
        # (i, k) and domain (j, l) sharing the same axis lengths.
        i, j, k, l, aux = namedoneto.((2, 2, 2, 2, 5), ("i", "j", "k", "l", "aux"))
        b = randn(elt, aux, i, k)
        # conj(b) * b with the non-conjugated copy's (i, k) relabeled to
        # (j, l) to form the operator-shaped Hermitian a ≈ X * X'.
        b_dom = replacedimnames(b, "i" => "j", "k" => "l")
        a = conj(b) * b_dom

        let X = gram_eigh_full(a, (i, k), (j, l))
            X_cod = replacedimnames(X, "j" => "i", "l" => "k")
            @test (j, l) ⊆ inds(X)
            @test X_cod * conj(X) ≈ a
        end

        let (X, Y) = gram_eigh_full_with_pinv(a, (i, k), (j, l))
            rank_name = only(setdiff(dimnames(X), ("j", "l")))
            @test rank_name == only(setdiff(dimnames(Y), ("j", "l")))
            X_cod = replacedimnames(X, "j" => "i", "l" => "k")
            @test X_cod * conj(X) ≈ a
            # Rename one rank dimension so `Y * X` contracts only on
            # the shared domain names `(j, l)` and leaves a
            # (rank × rank) named identity.
            fresh_rank = uniquename(rank_name)
            X_fresh = replacedimnames(X, rank_name => fresh_rank)
            YXmat = unname(Y * X_fresh, (rank_name, fresh_rank))
            @test YXmat ≈ LinearAlgebra.I(size(YXmat, 1))
        end
    end
    @testset "tr" begin
        i, j = Index.((2, 3))
        ip, jp = prime(i), prime(j)
        a = randn(elt, i, j, ip, jp)
        # The trace pairs (i, j) with (ip, jp), matching the dense matrix trace of the
        # matricized map.
        @test tr(a, (i, j), (ip, jp)) ≈ tr(reshape(unname(a, (i, j, ip, jp)), 6, 6))
        # The identity map traces to its (shared) fused dimension.
        @test tr(id(elt, (i, j), (ip, jp)), (i, j), (ip, jp)) ≈ 6
        # The operator form traces over its intrinsic codomain/domain split.
        op = operator(a, (name(i), name(j)), (name(ip), name(jp)))
        @test tr(op) ≈ tr(a, (i, j), (ip, jp))
    end
    @testset "project" begin
        i = Index(2)
        Sz = elt[0.5 0; 0 -0.5]
        # the three-argument form builds an operator from the codomain/domain split
        top = project(Sz, (prime(i),), (i,))
        @test eltype(top) === elt
        @test Set(dimnames(top)) == Set(name.((prime(i), i)))
        @test unname(top, (prime(i), i)) == Sz
        # `unchecked_project` skips the (for dense, always exact) verification
        @test unname(unchecked_project(Sz, (prime(i),), (i,)), (prime(i), i)) == Sz
        # the two-argument form builds a state (empty domain)
        v = elt[1, 0]
        s = project(v, (i,))
        @test dimnames(s) == [name(i)]
        @test unname(s, (i,)) == v
        # the empty-codomain form builds an all-domain tensor (mirror of the state)
        bra = project(v, (), (i,))
        @test dimnames(bra) == [name(i)]
        @test unname(bra, (i,)) == v
    end
    @testset "replacedimnames with index keys" begin
        i, j, k = namedoneto.((2, 3, 2), ("i", "j", "k"))
        a = randn(elt, i, j)
        # An `Index`-keyed pair relabels like the name-keyed pair rather than silently
        # no-opping, and the result stays an `ITensor` (not `NamedTensor{Any}`).
        @test dimnames(replacedimnames(a, i => k)) ==
            dimnames(replacedimnames(a, "i" => "k"))
        @test replacedimnames(a, i => k) isa typeof(a)
        # Mixed index/name keys and values are accepted.
        @test dimnames(replacedimnames(a, i => "k")) ==
            dimnames(replacedimnames(a, "i" => "k"))
        @test dimnames(replacedimnames(a, "i" => k)) ==
            dimnames(replacedimnames(a, "i" => "k"))
    end
    @testset "trivialrange on named ranges" begin
        i = Index(3)
        r = trivialrange(i)
        @test r isa Index
        @test length(r) == 1
        @test name(r) != name(i)
        rn = trivialrange(i, 4)
        @test rn isa Index
        @test length(rn) == 4
        @test name(rn) != name(i)
    end
end
