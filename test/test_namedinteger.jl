using ITensorBase: Named, NamedInteger, denamed, name, named, namedoneto
using Test: @test, @testset

@testset "Named integer" begin
    i = named(3, :i)
    @test i isa Named
    @test i isa NamedInteger
    @test denamed(i) ≡ 3
    @test name(i) ≡ :i
end

@testset "Named equality and hash invariant" begin
    # Equality and hashing are type-agnostic across named array types, following
    # Base's array convention (`[1, 2, 3] == 1:3` and they hash equally). A named
    # array and a named unit range with equal denamed values and names are equal, so
    # they must hash equally too.
    na = named([1, 2, 3], "x")
    nr = namedoneto(3, "x")
    @test na == nr
    @test hash(na) == hash(nr)
    # Differing value or name stays distinct.
    @test named([1, 2, 4], "x") != na
    @test named([1, 2, 3], "y") != na
    @test hash(named([1, 2, 3], "y")) != hash(na)
end
