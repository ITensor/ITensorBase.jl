using GradedArrays: U1, sectors
using ITensorBase: ITensorBase, Index, inds, prime, space
using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, isdual, project, project_aux, tryproject,
    tryproject_aux, unchecked_project, unchecked_project_aux
using TensorKitSectors: FermionNumber
using Test: @test, @test_throws, @testset

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

    # The rng-first flux forms (default eltype) accept both flat and split axes.
    @test length(inds(randn(rng, U1(1), (i, j)))) == 3
    @test length(inds(randn(rng, U1(1), (i,), (j,)))) == 3

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

    # `zeros`/`ones`/`fill` mirror `randn` (`fill` takes the value first). Each carries the
    # flux on an aux leg the same way.
    z = zeros(U1(1), (i, j))
    @test length(inds(z)) == 3
    @test eltype(zeros(elt, U1(1), (i, j))) == elt
    @test iszero(z)
    o = ones(elt, U1(1), (i, j))
    @test length(inds(o)) == 3
    @test only(sectors(space(only(setdiff(collect(inds(o)), [i, j]))))) == U1(1)
    fl = fill(elt(2), U1(1), (i,), (j,))
    @test length(inds(fl)) == 3
    @test eltype(fl) == elt

    # Empty codomain: every physical leg lives in the (dualized) domain, alongside the aux leg.
    e = randn(rng, elt, U1(1), (), (i, j))
    @test length(inds(e)) == 3
    auxe = only(setdiff(collect(inds(e)), [i, j]))
    @test isdual(auxe) && length(auxe) == 1 && only(sectors(space(auxe))) == U1(1)
    @test eltype(e) == elt
    @test length(inds(randn(rng, U1(1), (), (i, j)))) == 3
    @test length(inds(zeros(U1(1), (), (i, j)))) == 3
    @test length(inds(ones(elt, U1(1), (), (i,)))) == 2
    @test length(inds(fill(elt(2), U1(1), (), (j,)))) == 2
end

# `project_aux` and its siblings derive a named auxiliary leg carrying the operator's flux, so a
# charge-shifting operator stays symmetry-allowed instead of being projected away. Strict `project`
# instead projects into exactly the given indices and rejects a surplus axis.
@testset "project_aux derives a named auxiliary leg (eltype = $elt)" for elt in
    (
        Float64,
        ComplexF64,
    )
    s = Index([U1(0) => 1, U1(1) => 1]; tags = "s")
    cdag = elt[0 0; 1 0]   # raising operator, flux +1

    # without an auxiliary leg the charge-shifting operator has nothing to carry its flux
    @test iszero(unchecked_project(cdag, (prime(s),), (s,)))

    # strict `project` projects into exactly the given indices, so a surplus axis is an error
    @test_throws ArgumentError project(reshape(cdag, (2, 2, 1)), (prime(s),), (s,))

    # each `*_aux` verb derives the flux-canceling aux leg, whether given the physical rank or a
    # trailing length-1 slice axis
    @testset "$f" for f in (project_aux, tryproject_aux, unchecked_project_aux)
        for a in (cdag, reshape(cdag, (2, 2, 1)))
            op = f(a, (prime(s),), (s,))
            @test length(inds(op)) == 3
            @test !iszero(op)
            @test eltype(op) == elt
            aux = only(setdiff(collect(inds(op)), [prime(s), s]))
            @test length(aux) == 1
            @test isdual(aux)                          # dualized, in the domain
            @test only(sectors(space(aux))) == U1(1)   # carries the operator's flux
        end
    end
end
