using ITensorBase: ITensorBase
using Test: @test, @testset
@testset "Test exports" begin
    exports = [
        :ITensorBase, :AbstractNamedTensor, :NamedTensor, :AbstractITensor, :ITensor,
        :Index, :NamedUnitRange,
        :aligndims, :aligneddims, :apply, :codomainnames, :dimnames, :dimnametype,
        :domainnames,
        :inds, :mapinds, :named, :nameddims, :noprime, :operator, :prime, :replaceinds,
        :sim, :similar_operator, :state, :uniquename,
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
