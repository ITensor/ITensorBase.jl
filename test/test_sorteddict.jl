using ITensorBase: ITensorBase
using Test: @test, @testset

const SortedDict = ITensorBase.SortedDict

@testset "SortedDict" begin
    d = SortedDict{Symbol, Symbol}(:b => :y, :a => :x)
    @test collect(keys(d)) == [:a, :b]          # sorted on construction
    @test d[:a] == :x
    @test haskey(d, :b)
    @test !haskey(d, :c)
    @test get(d, :c, :default) == :default
    @test length(d) == 2

    d[:c] = :z                                   # insert keeps sort order
    @test collect(keys(d)) == [:a, :b, :c]
    @test d[:c] == :z

    d2 = copy(d)
    delete!(d2, :a)
    @test haskey(d, :a)                          # copy is independent
    @test !haskey(d2, :a)

    # order-independent equality + hash
    e1 = SortedDict{Symbol, Symbol}(:a => :x, :b => :y)
    e2 = SortedDict{Symbol, Symbol}(:b => :y, :a => :x)
    @test e1 == e2
    @test hash(e1) == hash(e2)
    @test isequal(e1, e2)

    @test length(SortedDict{Symbol, Symbol}()) == 0
    @test sort(collect(d)) == [:a => :x, :b => :y, :c => :z]
end
