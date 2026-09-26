using ITensorBase: Named, NamedArray, NamedInteger, NamedOneTo, name, unnamed
using Test: @test, @testset

@testset "Named integer" begin
    i = Named(3, :i)
    @test i isa Named
    @test i isa NamedInteger
    @test unnamed(i) ≡ 3
    @test name(i) ≡ :i
end

@testset "Named equality and hash invariant" begin
    # Equality and hashing are type-agnostic across named array types, following
    # Base's array convention (`[1, 2, 3] == 1:3` and they hash equally). A named
    # array and a named unit range with equal unnamed values and names are equal, so
    # they must hash equally too.
    na = NamedArray([1, 2, 3], "x")
    nr = NamedOneTo(3, "x")
    @test na == nr
    @test hash(na) == hash(nr)
    # Differing value or name stays distinct.
    @test NamedArray([1, 2, 4], "x") != na
    @test NamedArray([1, 2, 3], "y") != na
    @test hash(NamedArray([1, 2, 3], "y")) != hash(na)
end
