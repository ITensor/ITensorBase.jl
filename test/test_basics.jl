using BlockArrays: Block
using BlockSparseArrays: BlockSparseArray
using DiagonalArrays: δ, delta, diagview
using GradedUnitRanges: dual, gradedrange
using ITensorBase:
  ITensorBase,
  ITensor,
  Index,
  gettag,
  hastag,
  hasqns,
  inds,
  plev,
  prime,
  settag,
  tags,
  unsettag
using NamedDimsArrays: dename, name, named
using SparseArraysBase: oneelement
using SymmetrySectors: U1
using Test: @test, @test_broken, @test_throws, @testset

@testset "ITensorBase" begin
  @testset "Basics" begin
    elt = Float64
    i, j = Index.((2, 2))
    x = randn(elt, 2, 2)
    for a in (ITensor(x, i, j), ITensor(x, (i, j)))
      @test dename(a) == x
      @test plev(i) == 0
      @test plev(prime(i)) == 1
      @test length(tags(i)) == 0
      a′ = prime(a)
      @test dename(a′) == x
      @test issetequal(inds(a′), (prime(i), prime(j)))
    end

    @test_throws ErrorException ITensor(randn(elt, 2, 2), Index.((2, 3)))
    @test_throws ErrorException ITensor(randn(elt, 4), Index.((2, 2)))

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
  @testset "oneelement" begin
    i = Index(3)
    a = oneelement(i => 2)
    @test a isa ITensor
    @test ndims(a) == 1
    @test issetequal(inds(a), (i,))
    @test eltype(a) === Bool
    @test a[1] == 0
    @test a[2] == 1
    @test a[3] == 0

    i = Index(3)
    a = oneelement(Float32, i => 2)
    @test a isa ITensor
    @test ndims(a) == 1
    @test issetequal(inds(a), (i,))
    @test eltype(a) === Float32
    @test a[1] == 0
    @test a[2] == 1
    @test a[3] == 0
  end
  @testset "hasqns" begin
    i = Index(2)
    j = Index(2)
    a = ITensor(randn(2, 2), (i, j))
    @test !hasqns(i)
    @test !hasqns(j)
    @test !hasqns(a)

    r = gradedrange([U1(0) => 2, U1(1) => 2])
    d = BlockSparseArray{Float64}(r, dual(r))
    d[Block(1, 1)] = randn(2, 2)
    d[Block(2, 2)] = randn(2, 2)
    i = Index(r)
    j = Index(dual(r))
    a = ITensor(d, (i, j))
    @test hasqns(i)
    @test hasqns(j)
    @test hasqns(a)
  end
end
