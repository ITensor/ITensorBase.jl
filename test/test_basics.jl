using ITensorBase: ITensorBase, AbstractNamedTensor, ITensor, Index, IndexName, NamedTensor,
    commonind, commoninds, dimnametype, gettag, hascommoninds, hastag, id, inds, mapinds,
    name, named, noncommoninds, noprime, plev, prime, replaceinds, setplev, settag, sim,
    tags, trycommonind, trynoncommonind, unioninds, uniqueind, uniqueinds, uniquename,
    unname, unnamed, unsettag, uuid
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
        @test length(tags(i)) == 0
        a′ = mapinds(prime, a)
        @test unnamed(a′) == x
        @test issetequal(inds(a′), (prime(i), prime(j)))

        # The number of dimnames must match the array's `ndims`, and the dimnames are
        # passed as a single collection.
        @test_throws ArgumentError NamedTensor(randn(elt, 4), (:i, :j))
        @test_throws MethodError NamedTensor(randn(elt, 2, 2), :i, :j)

        # Passing indices as a tuple or vector builds the tensor, using only their names and
        # taking the space from the array. A single bare index still errors (it is ambiguous).
        i, j = Index.((2, 3))
        @test NamedTensor(randn(elt, 2, 3), (i, j)) isa ITensor
        @test ITensor(randn(elt, 2, 3), (i, j)) isa ITensor
        @test ITensor(randn(elt, 2, 3), [i, j]) isa ITensor
        t = ITensor(randn(elt, 2, 3), (i, j))
        @test issetequal(name.(inds(t)), name.((i, j)))
        @test_throws ArgumentError ITensor(randn(elt, 2), i)
        # The space is taken from the array, not from the index (a mismatched index dim is ignored).
        @test size(unnamed(ITensor(randn(elt, 2, 3), (i, Index(9))))) == (2, 3)
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
    @testset "dimnametype" begin
        i, j = Index.((2, 3))
        a = randn(Float64, i, j)
        @test a isa NamedTensor
        @test dimnametype(a) === IndexName
        @test dimnametype(typeof(a)) === IndexName
        @test dimnametype(NamedTensor{IndexName}) === IndexName
        # Unparameterized `NamedTensor` does not fix its dimname flavor, like `eltype(Array)`.
        @test dimnametype(NamedTensor) === Any
    end
    @testset "show" begin
        i = Index(2)
        @test sprint(show, "text/plain", i) ==
            "Index(length=2|id=$(first(string(uuid(i)), 8)))"

        i = settag(Index(2), "X", "Y")
        @test sprint(show, "text/plain", i) ==
            "Index(length=2|id=$(first(string(uuid(i)), 8))|X=>Y)"
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

        # `replaceinds` is a name-only synonym for the pair-based relabel.
        k, l = Index.((2, 3))
        a_r = replaceinds(a, i => k, j => l)
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
        @test name(trynoncommonind(a, b)) == name(i)

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
        @test isnothing(trynoncommonind(a, c))
    end
end
