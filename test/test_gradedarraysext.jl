using GradedArrays: U1, sectors
using ITensorBase: ITensorBase, Index, inds, space
using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, isdual
using TensorKitSectors: FermionNumber
using Test: @test, @testset

# The flux-canceling constructor mints an auxiliary `Index` carrying the requested charge and
# appends it to the domain, so an `ITensor` over graded (block-sparse) indices can be built
# with a nonzero total flux directly. Covers an abelian (U₁) and a fermionic sector.
@testset "GradedArraysExt flux-canceling constructor (eltype = $elt)" for elt in
    (
        Float64,
        ComplexF64,
    )
    rng = StableRNG(1234)

    i = Index([U1(0) => 1, U1(1) => 2]; tags = "i")
    j = Index([U1(0) => 2, U1(1) => 1]; tags = "j")

    # Flat form: all physical legs in the codomain, the aux the sole domain leg.
    a = randn(rng, elt, U1(1), (i, j))
    @test length(inds(a)) == 3
    @test i in inds(a)
    @test j in inds(a)
    aux = only(setdiff(collect(inds(a)), [i, j]))
    @test length(aux) == 1                     # multiplicity-1 aux leg
    @test isdual(aux)                          # dualized, in the domain
    @test only(sectors(space(aux))) == U1(1)   # carries the requested flux
    @test eltype(a) == elt

    # Map form: the aux is appended after the given domain leg.
    b = randn(rng, elt, U1(1), (i,), (j,))
    @test length(inds(b)) == 3
    auxb = only(setdiff(collect(inds(b)), [i, j]))
    @test isdual(auxb) && length(auxb) == 1 && only(sectors(space(auxb))) == U1(1)

    # A bare `TensorKitSectors.Sector` (fermionic) works as the flux.
    s = [
        Index([FermionNumber(0) => 2, FermionNumber(1) => 2]; tags = "s" => "$n") for
            n in 1:4
    ]
    t = randn(rng, elt, FermionNumber(2), (s[1], s[2], s[3], s[4]))
    @test length(inds(t)) == 5
    auxt = only(setdiff(collect(inds(t)), s))
    @test isdual(auxt) && length(auxt) == 1 &&
        only(sectors(space(auxt))) == FermionNumber(2)

    # `zeros` mirrors `randn`; the default eltype is `Float64`.
    z = zeros(U1(1), (i, j))
    @test length(inds(z)) == 3
    @test eltype(zeros(elt, U1(1), (i, j))) == elt
    @test iszero(z)
end
