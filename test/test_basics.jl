using ITensorBase: ITensorBase, ITensor, Index, gettag, inds, plev, prime, settag, unsettag
using DiagonalArrays: δ, delta, diagview
using NamedDimsArrays: unname
using Test: @test, @test_broken, @testset

@testset "ITensorBase" begin
  @testset "Basics" begin
    i, j = Index.((2, 2))
    x = randn(2, 2)
    for a in (
      ITensor(x, i, j),
      ITensor(x, (i, j)),
    )
      @test unname(a) == x
      @test plev(i) == 0
      @test plev(prime(i)) == 1
      a′ = prime(a)
      @test unname(a′) == x
      @test issetequal(inds(a′), (prime(i), prime(j)))
    end

    i = Index(2)
    i = settag(i, "X", "x")
    @test gettag(i, "X") == "x"
    i = unsettag(i, "X")
    @test isnothing(gettag(i, "X", nothing))
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
      @test diagview(unname(a)) == ones(2)
    end
  end
end
