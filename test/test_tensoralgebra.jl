using ITensorBase:
    ITensorBase, dimnames, inds, namedoneto, replacedimnames, uniquename, unname, unnamed
using LinearAlgebra: LinearAlgebra, norm
using MatrixAlgebraKit: left_null, left_orth, left_polar, lq_compact, lq_full, qr_compact,
    qr_full, right_null, right_orth, right_polar, svd_compact, svd_trunc
using StableRNGs: StableRNG
using TensorAlgebra.MatrixAlgebra: gram_eigh_full, gram_eigh_full_with_pinv
using TensorAlgebra: TensorAlgebra, contract, matricize, unmatricize
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

        # Test truncation.
        a = randn(elt, i, j, k, l)
        u, s, v = svd_trunc(a, (i, k), (j, l); trunc = (; maxrank = 2))
        @test u * s * v ≉ a
        @test size(s) == (2, 2)
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
end
