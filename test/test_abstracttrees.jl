using AbstractTrees: printnode
using ITensorBase: NamedTensor
using Test: @test, @testset

@testset "AbstractTrees" begin
    a = randn(3, 4)
    na = NamedTensor(a, ("i", "j"))
    @test sprint(printnode, na) == "{\"i\", \"j\"}"
end
