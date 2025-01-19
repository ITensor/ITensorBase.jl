using ITensorBase: ITensorBase, ITensor, Index, inds, plev, prime
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
end
