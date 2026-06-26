using ITensorBase: ITensorBase
using Test: @test, @testset
@testset "Test exports" begin
    exports = [
        :ITensorBase, :AbstractITensor, :ITensor, :Index, :NamedUnitRange,
        :aligndims, :apply, :codomainnames, :dimnames, :dimnametype, :domainnames,
        :inds, :named, :nameddims, :noprime, :operator, :prime, :similar_operator,
        :state, :uniquename,
    ]
    publics = [
        :name, :nametype, :replacedimnames, :setname, :unnamed, :unnamedtype,
        Symbol("@names"),
    ]
    if VERSION ≥ v"1.11-"
        exports = [exports; publics]
    end
    @test issetequal(names(ITensorBase), exports)
end
