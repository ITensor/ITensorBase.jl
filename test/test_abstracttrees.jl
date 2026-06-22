using AbstractTrees: printnode
using ITensorBase: nameddims
using Test: @test, @testset

@testset "AbstractTrees" begin
    a = randn(3, 4)
    na = nameddims(a, ("i", "j"))
    @test sprint(printnode, na) == "{\"i\", \"j\"}"
end
