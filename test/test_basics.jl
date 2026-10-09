using ITensorBase: ITensorBase, AbstractNamedTensor, ITensor, Index, IndexName, NamedTensor,
    commonind, commoninds, gettag, hascommoninds, hastag, id, inds, name, names, nametype,
    noncommonind, noncommoninds, noprime, operator, plev, prime, rename, setplev, settag,
    settags, sim, tags, trycommonind, trynoncommonind, tryuniqueind, unioninds, uniqueind,
    uniqueinds, uniquename, unname, unnamed, unsettag, uuid
using Test: @test, @test_broken, @test_throws, @testset
using UUIDs: UUID

@testset "ITensorBase" begin
    @testset "IndexName" begin
        n1 = IndexName(; uuid = UUID(0))
        n2 = IndexName(; uuid = UUID(0))
        @test n1 == n2
        @test isequal(n1, n2)
        @test hash(n1) ≡ hash(n2)

        n1 = IndexName(; uuid = UUID(0))
        n2 = IndexName(; uuid = UUID(1))
        @test n1 ≠ n2
        @test !isequal(n1, n2)
        @test n1 < n2
        @test isless(n1, n2)
        @test hash(n1) ≠ hash(n2)

        n1 = IndexName(; uuid = UUID(0), plev = 0)
        n2 = IndexName(; uuid = UUID(0), plev = 1)
        @test n1 ≠ n2
        @test !isequal(n1, n2)
        @test n1 < n2
        @test isless(n1, n2)
        @test hash(n1) ≠ hash(n2)

        for tagspec in (
                Dict(["X" => "Y"]), ["X" => "Y"], ("X" => "Y",), "X" => "Y",
                Dict([:X => :Y]), (:X => :Y,),
            )
            n = IndexName(; tags = tagspec)
            @test hastag(n, "X")
            @test hastag(n, :X)
            @test gettag(n, "X") == "Y"
            @test gettag(n, :X) == "Y"
            @test length(tags(n)) == 1
        end

        # Two-layer contract: public `tags` returns strings, internal stored layer uses Symbols.
        n = IndexName(; tags = "X" => "Y")
        @test tags(n) isa AbstractDict{<:AbstractString, <:AbstractString}
        @test tags(n)["X"] == "Y"
        @test gettag(n, "X") isa AbstractString
        @test ITensorBase.tags_stored(n)[:X] === :Y
    end
    @testset "uniquename" begin
        i = settag(prime(Index(2)), "X", "Y")
        # On an instance, only the uuid is fresh: tags and prime level are kept.
        n = uniquename(name(i))
        @test n != name(i)
        @test uuid(n) != uuid(name(i))
        @test plev(n) == plev(i)
        @test tags(n) == tags(i)
        # On the name type, a bare name: no tags, prime level zero.
        m = uniquename(IndexName)
        @test m isa IndexName
        @test plev(m) == 0
        @test isempty(tags(m))
        # On an `Index`, recurse into the name and keep its decoration.
        i′ = uniquename(i)
        @test name(i′) != name(i)
        @test plev(i′) == plev(i)
        @test tags(i′) == tags(i)
    end
    @testset "Index basics" begin
        i = Index(2)
        @test plev(i) == 0
        i = setplev(i, 2)
        @test plev(i) == 2

        i = Index(2)
        i = settag(i, "X", "x")
        @test hastag(i, "X")
        @test !hastag(i, "Y")
        @test gettag(i, "X") == "x"
        i = unsettag(i, "X")
        @test isnothing(gettag(i, "X", nothing))
        @test !hastag(i, "X")
        @test !hastag(i, "Y")

        i = Index(Base.OneTo(2))
        @test length(i) == 2
        @test length(i) isa Int
        @test unnamed(i) == 1:2
        @test plev(i) == 0
        @test length(tags(i)) == 0

        # An integer length is routed through `to_range` to a `Base.OneTo`, and an
        # explicit range is passed through unchanged.
        i = Index(3)
        @test unnamed(i) === Base.OneTo(3)
        i = Index(2:4)
        @test length(i) == 3
        @test unnamed(i) === 2:4

        i = settag(Index(2), "X", "Y")
        @test length(i) == 2
        @test hastag(i, "X")
        @test gettag(i, "X") == "Y"
        @test plev(i) == 0
        @test length(tags(i)) == 1
    end
    @testset "NamedTensor basics" begin
        elt = Float64
        i, j = Index.((2, 2))
        x = randn(elt, 2, 2)
        a = x[i, j]
        @test unnamed(a) == x
        @test plev(i) == 0
        @test plev(prime(i)) == 1
        # `plinc` lives on the index, never on a tensor, so a tensor's second argument is
        # always a selection. `prime(i, n)` is how a level-`n` index gets named.
        @test plev(prime(i, 2)) == 2
        @test plev(prime(name(i), 3)) == 3
        @test prime(i, 1) == prime(i)
        @test prime(i, 0) == i
        @test prime(prime(i), 2) == prime(i, 3)
        @test length(tags(i)) == 0
        a′ = rename(prime, a)
        @test unnamed(a′) == x
        @test issetequal(inds(a′), (prime(i), prime(j)))

        # The number of names must match the array's `ndims`, and the names are
        # passed as a single collection.
        @test_throws ArgumentError NamedTensor(randn(elt, 4), (:i, :j))
        @test_throws MethodError NamedTensor(randn(elt, 2, 2), :i, :j)

        # Passing indices as a tuple or vector builds the tensor from their names, keeping only
        # the names. A single bare index still errors (it is ambiguous).
        i, j = Index.((2, 3))
        @test NamedTensor(randn(elt, 2, 3), (i, j)) isa ITensor
        @test ITensor(randn(elt, 2, 3), (i, j)) isa ITensor
        @test ITensor(randn(elt, 2, 3), [i, j]) isa ITensor
        t = ITensor(randn(elt, 2, 3), (i, j))
        @test issetequal(name.(inds(t)), name.((i, j)))
        @test_throws ArgumentError ITensor(randn(elt, 2), i)
        # A dimension given as an index asserts a space, which has to match the array's axis.
        @test_throws ArgumentError ITensor(randn(elt, 2, 3), (i, Index(9)))
        @test_throws ArgumentError NamedTensor(randn(elt, 2, 3), (i, Index(9)))
        # A bare name asserts no space, so it is unaffected by the check.
        @test ITensor(randn(elt, 2, 3), (name(i), name(Index(9)))) isa ITensor

        # The codomain/domain form takes the two dimension groups separately, with the same
        # space check on each group and the same rejection of a lone index.
        @test NamedTensor(randn(elt, 2, 3), (i,), (j,)) isa ITensor
        @test names(ITensor(randn(elt, 2, 3), (i,), (j,))) == name.([i, j])
        # Dense storage carries no split of its own, so either grouping is accepted.
        @test ITensor(randn(elt, 2, 3), (), (i, j)) isa ITensor
        @test_throws ArgumentError ITensor(randn(elt, 2, 3), i, (j,))
        @test_throws ArgumentError ITensor(randn(elt, 2, 3), (i,), j)
        @test_throws ArgumentError ITensor(randn(elt, 2, 3), (i,), (Index(9),))
        @test_throws ArgumentError ITensor(randn(elt, 2), (i,), (j,))
        # The other supported constructions: index the array (inherit the space from the
        # indices), or attach only the names (take the space from the array).
        @test randn(elt, 2, 3)[i, j] isa ITensor
        @test ITensor(randn(elt, 2, 3), name.((i, j))) isa ITensor

        i, j = Index.((3, 4))
        a = randn(elt, i, j)
        a′ = Array(a)
        @test eltype(a′) === elt
        @test a′ isa Matrix{elt}
        @test a′ == unnamed(a)
        # `Array` returns a fresh copy, not the storage object itself.
        @test a′ !== unnamed(a)

        i, j = Index.((3, 4))
        a = randn(elt, i, j)
        for a′ in (Array{Float32}(a), Matrix{Float32}(a))
            @test eltype(a′) === Float32
            @test a′ isa Matrix{Float32}
            @test a′ == Float32.(unnamed(a))
        end

        i, j, k = Index.((2, 2, 2))
        a = randn(elt, i, j, k)
        b = randn(elt, k, i, j)
        copyto!(a, b)
        @test a == b
        @test unnamed(a) == unname(b, (i, j, k))
        @test unnamed(a) == permutedims(unnamed(b), (2, 3, 1))
    end
    @testset "nametype" begin
        i, j = Index.((2, 3))
        a = randn(Float64, i, j)
        @test a isa NamedTensor
        @test nametype(a) === IndexName
        @test nametype(typeof(a)) === IndexName
        @test nametype(NamedTensor{IndexName}) === IndexName
        # An operator reports the dimname flavor of its underlying tensor.
        op = operator(a, (name(i),), (name(j),))
        @test nametype(op) === IndexName
        @test nametype(typeof(op)) === IndexName
        # Unparameterized `NamedTensor` does not fix its dimname flavor, like `eltype(Array)`.
        @test nametype(NamedTensor) === Any
    end
    @testset "show" begin
        i = Index(2)
        @test sprint(show, "text/plain", i) ==
            "Index(2|id=$(first(string(uuid(i)), 8)))"

        i = settag(Index(2), "X", "Y")
        @test sprint(show, "text/plain", i) ==
            "Index(2|id=$(first(string(uuid(i)), 8))|X=>Y)"
    end
    @testset "selected-index manipulation" begin
        elt = Float64
        i, j, k = Index.((2, 3, 4))
        a = randn(elt, i, j, k)

        # `sim` is absent here because it mints a fresh id on every call, so two separate calls
        # never compare equal; it is checked on its own below. Each function is given a tensor
        # it actually changes, so the comparisons are not trivially true.
        for (f, b) in ((prime, a), (noprime, prime(a)))
            bi, _, bk = inds(b)
            # A lone index or index name stands for the one-element collection. An `Index` is a
            # `NamedUnitRange`, so without its own method it would be read as a range of integers.
            @test f(b, bi) == f(b, (bi,))
            @test f(b, name(bi)) == f(b, (bi,))
            # Any collection of indices works.
            @test f(b, [bi, bk]) == f(b, (bi, bk))
            # Selecting every index agrees with the whole-tensor form.
            @test f(b, inds(b)) == f(b)
        end

        # Properties shared by all three, each read off a single call.
        for f in (prime, noprime, sim)
            # The unselected indices are left alone.
            @test inds(f(a, (i, k)))[2] == j
            # Selecting none is a no-op, as is naming an index the tensor does not have.
            @test f(a, ()) == a
            @test f(a, Index(5)) == a
            # Relabeling is name-only, so the data is untouched.
            @test unnamed(f(a, i)) == unnamed(a)
        end

        # `prime` and `noprime` select on the full name, prime level included, so a primed index
        # has to be named as such. The predicate form is the way to avoid spelling it out.
        a′ = prime(a, (i, j))
        @test issetequal(inds(a′), (prime(i), prime(j), k))
        @test noprime(a′, i) == a′
        @test noprime(a′, prime(i)) == prime(a, j)
        @test prime(n -> plev(n) == 0, a′) == prime(a)
        # A predicate that selects every primed index is just `noprime(a)`, so test one that
        # leaves a primed index behind.
        a′′ = prime(a′, (prime(i),))
        @test issetequal(inds(a′′), (prime(prime(i)), prime(j), k))
        @test noprime(n -> plev(n) == 2, a′′) == prime(a, j)
        # Naming the level-2 index directly is the alternative to the predicate.
        @test noprime(a′′, prime(i, 2)) == prime(a, j)

        # `prime` alone takes a level increment, before the selection. `noprime` and `sim`
        # have no level to count, so they take no such argument.
        @test prime(a, 2) == prime(prime(a))
        @test inds(prime(a, 2, i)) == [prime(i, 2), j, k]
        @test inds(prime(a, 2, (i, k))) == [prime(i, 2), j, prime(k, 2)]
        @test prime(a, 2, name(i)) == prime(a, 2, i)
        @test prime(a, 1) == prime(a)
        @test prime(a, 0) == a
        # Stepping up twice matches one increment of two only when the second step names the
        # index at the level it has reached, since selection includes the prime level.
        @test prime(a, 2, i) == prime(prime(a, i), prime(i))
        # A negative increment lowers the level, which is what an `unprime` would do.
        @test prime(prime(a, 2), -2) == a
        @test prime(prime(a, 2, i), -1, prime(i, 2)) == prime(a, i)
        # The predicate form takes the increment too, trailing the predicate.
        @test prime(n -> plev(n) == 0, a, 2) == prime(a, 2)
        @test prime(x -> x == i, a, 2) == prime(a, 2, i)
        @test prime(n -> plev(n) == 0, a, 1) == prime(n -> plev(n) == 0, a)

        # The predicate sees the indices, not the names, so index-level accessors work.
        i_t = settags(i, "Site")
        a_t = randn(elt, i_t, j, k)
        @test prime(x -> hastag(x, "Site"), a_t) == prime(a_t, i_t)

        # `sim` mints a fresh id for the selected index only.
        a_s = sim(a, i)
        @test inds(a_s)[1] != i
        @test issetequal(inds(a_s)[2:3], (j, k))
        @test unnamed(a_s) == unnamed(a)
    end

    @testset "whole-tensor index manipulation" begin
        elt = Float64
        i, j = Index.((2, 3))
        a = randn(elt, i, j)

        # `prime`/`noprime` relabel every index name-only, leaving the data untouched.
        a′ = prime(a)
        @test unnamed(a′) == unnamed(a)
        @test issetequal(inds(a′), (prime(i), prime(j)))
        @test noprime(a′) == a
        @test issetequal(inds(noprime(prime(a′))), (i, j))

        # `rename` takes index-keyed pairs, relabeling name-only.
        k, l = Index.((2, 3))
        a_r = rename(a, i => k, j => l)
        @test unnamed(a_r) == unnamed(a)
        @test issetequal(inds(a_r), (k, l))

        # `sim` mints fresh ids, so no index of `sim(a)` matches an index of `a`, while the
        # data, lengths, tags, and prime levels are preserved.
        a_s = sim(a)
        @test unnamed(a_s) == unnamed(a)
        @test !any(in(inds(a)), inds(a_s))
        @test issetequal(length.(inds(a_s)), length.(inds(a)))

        i2 = settag(prime(Index(2)), "X", "Y")
        @test sim(i2) != i2
        @test plev(sim(i2)) == 1
        @test gettag(sim(i2), "X") == "Y"
        @test length(sim(i2)) == 2
    end
    @testset "rank-0 similar" begin
        elt = Float64
        i, j = Index.((2, 3))
        a = randn(elt, i, j)

        # `similar(a, ())` mints a scalar (0-dim) tensor on `a`'s backend and element type.
        s = similar(a, ())
        @test s isa AbstractNamedTensor
        @test eltype(s) === elt
        @test isempty(inds(s))
        @test ndims(unnamed(s)) == 0
        fill!(s, 1)
        @test unnamed(s)[] == 1

        s32 = similar(a, Float32, ())
        @test eltype(s32) === Float32
        @test isempty(inds(s32))
    end
    @testset "name-based index-set algebra" begin
        elt = Float64
        i, j, k = Index.((2, 3, 4))
        a = randn(elt, i, j)
        b = randn(elt, j, k)

        namesof(is) = name.(is)

        @test namesof(commoninds(a, b)) == namesof([j])
        @test namesof(uniqueinds(a, b)) == namesof([i])
        @test namesof(unioninds(a, b)) == namesof([i, j, k])
        @test namesof(noncommoninds(a, b)) == namesof([i, k])
        @test hascommoninds(a, b)

        @test name(commonind(a, b)) == name(j)
        @test name(uniqueind(a, b)) == name(i)
        @test name(trycommonind(a, b)) == name(j)
        @test name(tryuniqueind(a, b)) == name(i)

        # The symmetric-difference single: the one index not shared across both tensors.
        e = randn(elt, i, j, k)
        @test name(noncommonind(a, e)) == name(k)
        @test name(trynoncommonind(a, e)) == name(k)

        # No shared index.
        c = randn(elt, k)
        @test isempty(commoninds(a, c))
        @test !hascommoninds(a, c)
        @test isnothing(trycommonind(a, c))

        # `commonind`/`uniqueind` error unless there is exactly one; the `try*` forms
        # return `nothing` instead.
        d = randn(elt, i, j)
        @test_throws ArgumentError commonind(a, d)
        @test isnothing(trycommonind(a, d))
        @test_throws ArgumentError commonind(a, c)
        @test_throws ArgumentError uniqueind(a, c)
        @test isnothing(tryuniqueind(a, c))

        # `noncommonind` errors / `trynoncommonind` is `nothing` unless there is exactly one
        # non-shared index: two here (symdiff is `[i, k]`), zero for identical index sets.
        @test_throws ArgumentError noncommonind(a, b)
        @test isnothing(trynoncommonind(a, b))
        @test_throws ArgumentError noncommonind(a, d)
        @test isnothing(trynoncommonind(a, d))
    end
end
