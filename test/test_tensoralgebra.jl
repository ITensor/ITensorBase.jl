using ITensorBase:
    ITensorBase, Index, NamedOneTo, id, inds, name, operator, prime, rename, unname, unnamed
using LinearAlgebra: norm, tr
using MatrixAlgebraKit: left_null, left_orth, left_polar, lq_compact, lq_full, qr_compact,
    qr_full, right_null, right_orth, right_polar, svd_compact, svd_trunc, svd_vals
using StableRNGs: StableRNG
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
        i = NamedOneTo(2, "i")
        j = NamedOneTo(2, "j")
        k = NamedOneTo(2, "k")
        na1 = randn(elt, i, j)
        na2 = randn(elt, j, k)
        na_dest = na1 * na2
        @test eltype(na_dest) ≡ elt
        @test unname(na_dest, (i, k)) ≈ unnamed(na1) * unnamed(na2)
    end
    @testset "matricize" begin
        i, j, k, l = NamedOneTo.((2, 3, 4, 5), ("i", "j", "k", "l"))
        na = randn(elt, i, j, k, l)
        # The two dimension groups fuse into the rows and the columns of an unnamed matrix.
        m = matricize(na, (k, i), (j, l))
        @test m isa AbstractMatrix{elt}
        @test m ≈ reshape(
            unname(na, (k, i, j, l)),
            (
                length(k) * length(i),
                length(j) * length(l),
            )
        )
        # Groups may be any iterable of dimensions, not only tuples (no `Tuple` wrapping
        # needed).
        @test matricize(na, [k, i], [j, l]) == m
    end
    @testset "unmatricize" begin
        i, j, k, l = NamedOneTo.((2, 3, 4, 5), ("i", "j", "k", "l"))
        m = randn(elt, length(k) * length(i), length(j) * length(l))
        # An unnamed matrix splits back into a named tensor over the given codomain and
        # domain indices.
        na_split = unmatricize(m, (k, i), (j, l))
        @test unname(na_split, ("k", "i", "j", "l")) ≈
            reshape(m, (unnamed(k), unnamed(i), unnamed(j), unnamed(l)))
        # Round trip through the matrix.
        na = randn(elt, i, j, k, l)
        @test unmatricize(matricize(na, (k, i), (j, l)), (k, i), (j, l)) ≈ na
    end
    @testset "directsum" begin
        i = NamedOneTo(2, "i")            # shared index, carried through unchanged
        j1, j2 = NamedOneTo.((2, 3), ("j1", "j2"))
        k1, k2 = NamedOneTo.((2, 3), ("k1", "k2"))
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
        o1, o2 = NamedOneTo.((5, 5), ("o1", "o2"))
        s2 = directsum((o1, o2), a => (j1, k1), b => (j2, k2))
        @test issetequal(inds(s2), (i, o1, o2))
        @test unname(s2, (i, o1, o2)) == ref
        # A single summed dimension.
        u1, u2 = NamedOneTo.((2, 3), ("u1", "u2"))
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
                i, j, k, l = NamedOneTo.((2, 2, 2, 2), ("i", "j", "k", "l"))
                rng = StableRNG(123)
                a = randn(rng, $elt, (i, j, k, l))
                fa = $f(a, (j, l), (k, i))
                m = matricize(a, (j, l), (k, i))
                fm = matricize(fa, (j, l), (k, i))
                @test fm ≈ $f(m)
            end
        end
    end
    @testset "qr/lq" begin
        dims = (2, 2, 2, 2)
        i, j, k, l = NamedOneTo.(dims, ("i", "j", "k", "l"))

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
        i, j, k, l = NamedOneTo.(dims, ("i", "j", "k", "l"))

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
        i, j, k, l = NamedOneTo.(dims, ("i", "j", "k", "l"))

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
        @test Set(names(top)) == Set(name.((prime(i), i)))
        @test unname(top, (prime(i), i)) == Sz
        # `unchecked_project` skips the (for dense, always exact) verification
        @test unname(unchecked_project(Sz, (prime(i),), (i,)), (prime(i), i)) == Sz
        # the two-argument form builds a state (empty domain)
        v = elt[1, 0]
        s = project(v, (i,))
        @test names(s) == [name(i)]
        @test unname(s, (i,)) == v
        # the empty-codomain form builds an all-domain tensor (mirror of the state)
        bra = project(v, (), (i,))
        @test names(bra) == [name(i)]
        @test unname(bra, (i,)) == v
    end
    @testset "rename with index keys" begin
        i, j, k = NamedOneTo.((2, 3, 2), ("i", "j", "k"))
        a = randn(elt, i, j)
        # An `Index`-keyed pair relabels like the name-keyed pair rather than silently
        # no-opping, and the result stays an `ITensor` (not `NamedTensor{Any}`).
        @test names(rename(a, i => k)) ==
            names(rename(a, "i" => "k"))
        @test rename(a, i => k) isa typeof(a)
        # Mixed index/name keys and values are accepted.
        @test names(rename(a, i => "k")) ==
            names(rename(a, "i" => "k"))
        @test names(rename(a, "i" => k)) ==
            names(rename(a, "i" => "k"))
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
