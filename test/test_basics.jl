using ITensorBase: ITensorBase, ITensor, Index, IndexName, dimnametype, gettag, hastag, id,
    inds, mapinds, name, named, plev, prime, setplev, settag, tags, unname, unnamed,
    unsettag
using Test: @test, @test_broken, @test_throws, @testset
using UUIDs: UUID

@testset "ITensorBase" begin
    @testset "IndexName" begin
        n1 = IndexName(; id = UUID(0))
        n2 = IndexName(; id = UUID(0))
        @test n1 == n2
        @test isequal(n1, n2)
        @test hash(n1) ≡ hash(n2)

        n1 = IndexName(; id = UUID(0))
        n2 = IndexName(; id = UUID(1))
        @test n1 ≠ n2
        @test !isequal(n1, n2)
        @test n1 < n2
        @test isless(n1, n2)
        @test hash(n1) ≠ hash(n2)

        n1 = IndexName(; id = UUID(0), plev = 0)
        n2 = IndexName(; id = UUID(0), plev = 1)
        @test n1 ≠ n2
        @test !isequal(n1, n2)
        @test n1 < n2
        @test isless(n1, n2)
        @test hash(n1) ≠ hash(n2)

        for tagspec in (Dict(["X" => "Y"]), ["X" => "Y"], ("X" => "Y",), "X" => "Y")
            n = IndexName(; tags = tagspec)
            @test hastag(n, "X")
            @test gettag(n, "X") == "Y"
            @test length(tags(n)) == 1
        end
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
        @test length(i) == named(2, name(i))
        @test unnamed(length(i)) == 2
        @test unnamed(i) == 1:2
        @test plev(i) == 0
        @test length(tags(i)) == 0

        i = settag(Index(2), "X", "Y")
        @test unnamed(length(i)) == 2
        @test hastag(i, "X")
        @test gettag(i, "X") == "Y"
        @test plev(i) == 0
        @test length(tags(i)) == 1
    end
    @testset "ITensor basics" begin
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
        @test_throws ArgumentError ITensor(randn(elt, 4), Index.((2, 2)))
        @test_throws MethodError ITensor(randn(elt, 2, 2), Index(2), Index(2))

        i, j = Index.((3, 4))
        a = randn(elt, i, j)
        a′ = Array(a)
        @test eltype(a′) === elt
        @test a′ isa Matrix{elt}
        @test a′ == unnamed(a)

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
        @test a isa ITensor
        @test dimnametype(a) === IndexName
        @test dimnametype(typeof(a)) === IndexName
        @test dimnametype(ITensor{IndexName}) === IndexName
        # Unparameterized `ITensor` does not fix its dimname flavor, like `eltype(Array)`.
        @test dimnametype(ITensor) === Any
    end
    @testset "show" begin
        i = Index(2)
        @test sprint(show, "text/plain", i) ==
            "Index(length=2|id=$(first(string(id(i)), 8)))"

        i = settag(Index(2), "X", "Y")
        @test sprint(show, "text/plain", i) ==
            "Index(length=2|id=$(first(string(id(i)), 8))|\"X\"=>\"Y\")"
    end
end
