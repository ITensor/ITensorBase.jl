using ITensorBase: ITensorBase, ITensor, Index, IndexName, gettag, hastag, plev, prime,
    setplev, settag, tags, unsettag
using NamedDimsArrays: dename, denamed, inds, mapinds, name, named
using LinearAlgebra: factorize
using Test: @test, @test_broken, @test_throws, @testset

const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "ITensorBase" begin
    @testset "IndexName" begin
        n1 = IndexName(; id = UInt64(0))
        n2 = IndexName(; id = UInt64(0))
        @test n1 == n2
        @test isequal(n1, n2)
        @test hash(n1) ≡ hash(n2)

        n1 = IndexName(; id = UInt64(0))
        n2 = IndexName(; id = UInt64(1))
        @test n1 ≠ n2
        @test !isequal(n1, n2)
        @test n1 < n2
        @test isless(n1, n2)
        @test hash(n1) ≠ hash(n2)

        n1 = IndexName(; id = UInt64(0), plev = 0)
        n2 = IndexName(; id = UInt64(0), plev = 1)
        @test n1 ≠ n2
        @test !isequal(n1, n2)
        @test n1 < n2
        @test isless(n1, n2)
        @test hash(n1) ≠ hash(n2)
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
        @test denamed(length(i)) == 2
        @test denamed(i) == 1:2
        @test plev(i) == 0
        @test length(tags(i)) == 0

        for i in (
                Index(2; tags = Dict(["X" => "Y"])),
                Index(2; tags = ["X" => "Y"]),
                Index(2; tags = ("X" => "Y",)),
                Index(2; tags = "X" => "Y"),
            )
            @test Int(length(i)) == 2
            @test hastag(i, "X")
            @test gettag(i, "X") == "Y"
            @test plev(i) == 0
            @test length(tags(i)) == 1
        end

    end
    @testset "ITensor basics" begin
        elt = Float64
        i, j = Index.((2, 2))
        x = randn(elt, 2, 2)
        a = x[i, j]
        @test denamed(a) == x
        @test plev(i) == 0
        @test plev(prime(i)) == 1
        @test length(tags(i)) == 0
        a′ = mapinds(prime, a)
        @test denamed(a′) == x
        @test issetequal(inds(a′), (prime(i), prime(j)))

        # For now, the `ITensor` constructor is strict and only accepts a collection of
        # `IndexName` as dimnames.
        @test_throws ArgumentError ITensor(randn(elt, 2, 2), Index.((2, 2)))
        @test_throws ArgumentError ITensor(randn(elt, 2, 2), Index.((2, 3)))
        @test_throws ArgumentError ITensor(randn(elt, 4), Index.((2, 2)))
        @test_throws MethodError ITensor(randn(elt, 2, 2), Index(2), Index(2))

        i, j = Index.((3, 4))
        a = randn(elt, i, j)
        a′ = Array(a)
        @test eltype(a′) === elt
        @test a′ isa Matrix{elt}
        @test a′ == denamed(a)

        i, j = Index.((3, 4))
        a = randn(elt, i, j)
        for a′ in (Array{Float32}(a), Matrix{Float32}(a))
            @test eltype(a′) === Float32
            @test a′ isa Matrix{Float32}
            @test a′ == Float32.(denamed(a))
        end

        i, j, k = Index.((2, 2, 2))
        a = randn(elt, i, j, k)
        b = randn(elt, k, i, j)
        copyto!(a, b)
        @test a == b
        @test denamed(a) == dename(b, (i, j, k))
        @test denamed(a) == permutedims(denamed(b), (2, 3, 1))
    end
    @testset "show" begin
        id = rand(UInt64)
        i = Index(2; id)
        @test sprint(show, "text/plain", i) == "Index(length=2|id=$(id % 1000))"

        id = rand(UInt64)
        i = Index(2; id, tags = ["X" => "Y"])
        @test sprint(show, "text/plain", i) == "Index(length=2|id=$(id % 1000)|\"X\"=>\"Y\")"
    end
    @testset "factorize" for elt in elts
        i = Index(2)
        j = Index(2)
        a = randn(elt, i, j)
        x, y = factorize(a, (i,))
        @test a ≈ x * y
        @test x isa ITensor
        @test y isa ITensor
        @test i ∈ inds(x)
        @test j ∈ inds(y)
        @test eltype(x) === elt
        @test eltype(y) === elt
        @test Int.(Tuple(size(x))) == (2, 2)
        @test Int.(Tuple(size(y))) == (2, 2)

        i = Index(2)
        j = Index(2)
        a = randn(elt, i) * randn(elt, j)
        for kwargs in ((; rtol = 1.0e-2), (; cutoff = 1.0e-2))
            x, y = factorize(a, (i,); kwargs...)
            @test a ≈ x * y
            @test Int.(Tuple(size(x))) == (2, 1)
            @test Int.(Tuple(size(y))) == (1, 2)
        end
    end
end
