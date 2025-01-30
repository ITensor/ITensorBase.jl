using ITensorBase:
  ITensorBase, ITensor, Index, gettag, hastag, inds, plev, prime, settag, tags, unsettag
using DiagonalArrays: δ, delta, diagview
using NamedDimsArrays: dename, name, named
using Test: @test, @test_broken, @testset

@testset "ITensorBase" begin
  @testset "Basics" begin
    i, j = Index.((2, 2))
    x = randn(2, 2)
    for a in (ITensor(x, i, j), ITensor(x, (i, j)))
      @test dename(a) == x
      @test plev(i) == 0
      @test plev(prime(i)) == 1
      @test length(tags(i)) == 0
      a′ = prime(a)
      @test dename(a′) == x
      @test issetequal(inds(a′), (prime(i), prime(j)))
    end

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
    @test dename(length(i)) == 2
    @test dename(i) == 1:2
    @test plev(i) == 0
    @test length(tags(i)) == 0
  end
  @testset "delta" begin
    i, j = Index.((2, 2))
    for a in (
      delta(i, j),
      delta(Bool, i, j),
      delta((i, j)),
      delta(Bool, (i, j)),
      δ(i, j),
      δ(Bool, i, j),
      δ((i, j)),
      δ(Bool, (i, j)),
    )
      @test eltype(a) === Bool
      # TODO: Fix this.
      @test_broken diagview(a)
      @test diagview(dename(a)) == ones(2)
    end
  end
end
