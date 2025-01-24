using ITensorBase: ITensorBase, ITensor, Index, gettag, inds, plev, prime, settag, unsettag
using NamedDimsArrays: unname
using Test: @test, @testset

@testset "ITensorBase" begin
  i, j = Index.((2, 2))
  x = randn(2, 2)
  a = ITensor(x, (i, j))
  @test unname(a) == x
  @test plev(i) == 0
  @test plev(prime(i)) == 1
  a′ = prime(a)
  @test unname(a′) == x
  @test issetequal(inds(a′), (prime(i), prime(j)))

  i = Index(2)
  i = settag(i, "X", "x")
  @test gettag(i, "X") == "x"
  i = unsettag(i, "X")
  @test isnothing(gettag(i, "X", nothing))
end
