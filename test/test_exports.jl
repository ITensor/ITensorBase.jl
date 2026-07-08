using ITensorBase: ITensorBase
using Test: @test, @testset
@testset "Test exports" begin
    exports = [
        :ITensorBase, :AbstractNamedTensor, :NamedTensor, :AbstractITensor, :ITensor,
        :Index, :NamedUnitRange,
        :aligndims, :aligneddims, :apply, :codomainnames, :commonind, :commoninds,
        :dimnames, :dimnametype, :domainnames, :hascommoninds,
        :inds, :mapinds, :named, :nameddims, :noncommoninds, :noprime, :operator,
        :prime,
        :replaceinds, :sim, :similar_operator, :state, :trycommonind, :trynoncommonind,
        :uniqueind, :uniqueinds, :unioninds, :uniquename,
    ]
    publics = [
        :IndexName, :name, :nametype, :replacedimnames, :setname, :unnamed,
        :unnamedtype,
        Symbol("@names"),
    ]
    if VERSION ≥ v"1.11-"
        exports = [exports; publics]
    end
    @test issetequal(names(ITensorBase), exports)
end
