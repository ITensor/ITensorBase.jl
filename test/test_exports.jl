using ITensorBase: ITensorBase
using Test: @test, @testset
@testset "Test exports" begin
    exports = [
        :ITensorBase, :ITensor, :Index, :NamedDimsArray, :aligndims, :dimnametype,
        :named, :nameddims, :operator, :similar_operator,
    ]
    publics = [:to_inds, Symbol("@names")]
    if VERSION ≥ v"1.11-"
        exports = [exports; publics]
    end
    @test issetequal(names(ITensorBase), exports)
end
