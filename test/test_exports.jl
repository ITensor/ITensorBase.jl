using ITensorBase: ITensorBase
using Test: @test, @testset
@testset "Test exports" begin
    exports = [
        :ITensorBase, :AbstractNamedTensor, :NamedTensor, :AbstractITensor, :ITensor,
        :Index, :NamedUnitRange,
        :aligndims, :aligneddims, :apply, :codomainnames, :commonind, :commoninds,
        :dimnames, :dimnametype, :domainnames, :hascommoninds, :id,
        :inds, :mapinds, :named, :nameddims, :noncommonind, :noncommoninds, :noprime,
        :operator,
        :prime,
        :replaceinds, :sim, :similar_operator, :state, :trycommonind, :trynoncommonind,
        :tryuniqueind, :uniqueind, :uniqueinds, :unioninds, :uniquename,
    ]
    publics = [
        :IndexName, :name, :nametype, :replacedimnames, :setname, :space, :unnamed,
        :unnamedtype,
        :decoration, :emptytags, :gettag, :gettags, :hastag, :plev, :settags, :tags,
        :unsettags,
        Symbol("@names"),
    ]
    if VERSION ≥ v"1.11-"
        exports = [exports; publics]
    end
    @test issetequal(names(ITensorBase), exports)
end
