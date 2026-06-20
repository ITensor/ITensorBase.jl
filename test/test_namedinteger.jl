using ITensorBase: AbstractNamedInteger, NamedInteger, denamed, name, named
using Test: @test, @testset

@testset "Named integer" begin
    i = named(3, :i)
    @test i isa NamedInteger
    @test i isa AbstractNamedInteger
    @test denamed(i) ≡ 3
    @test name(i) ≡ :i
end
