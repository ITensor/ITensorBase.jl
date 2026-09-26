using GradedArrays: U1, sectors
using ITensorBase: ITensorBase, ITensor, Index, align, inds, prime, space, unnamed
using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, dual, isdual, matricize, project, project_aux,
    tryproject, tryproject_aux, unchecked_project, unchecked_project_aux, unmatricize
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

# Broadcasting over graded (GradedArrays.jl) indices routes the named expression through the
# `GradedArray` / matricized `FusedGradedMatrix` backend. Linear combinations add block-wise; a sum
# flattens all-codomain, so a within-split reorder is compared at a common split via `align`.
@testset "GradedArraysExt broadcasting (eltype = $elt)" for elt in (Float64, ComplexF64)
    rng = StableRNG(1234)
    i = Index([U1(0) => 2, U1(1) => 3]; tags = "i")
    j = Index([U1(0) => 1, U1(1) => 2]; tags = "j")
    k = Index([U1(-1) => 1, U1(0) => 2]; tags = "k")

    # Flat form (all-codomain, `GradedArray`-backed).
    a = randn(rng, elt, i, j)
    b = randn(rng, elt, i, j)
    @test unnamed(a .+ b) ≈ unnamed(a) + unnamed(b)
    @test unnamed(2 .* a) ≈ 2 * unnamed(a)
    @test unnamed(a .- 3 .* b) ≈ unnamed(a) - 3 * unnamed(b)

    # Map form (codomain/domain split, matricized `FusedGradedMatrix` storage).
    m = randn(rng, elt, (i,), (j,))
    n = randn(rng, elt, (i,), (j,))
    @test unnamed(m .+ n) ≈ unnamed(m) + unnamed(n)

    # Within-split reorder still adds correctly (the sum is all-codomain, compared via `align`).
    mr1 = randn(rng, elt, (i, j), (k,))
    mr2 = randn(rng, elt, (j, i), (k,))
    @test unnamed(align(mr1 .+ mr2, (i, j), (k,))) ≈
        unnamed(mr1) + unnamed(align(mr2, (i, j), (k,)))
end

# `matricize` fuses a tensor's codomain/domain split into an unnamed matrix and `unmatricize`
# splits one back out over named indices. The domain index is stored dualized while it is given
# to `unmatricize` codomain-facing, so a graded backend is where that convention is visible.
@testset "GradedArraysExt matricize/unmatricize (eltype = $elt)" for elt in
    (
        Float64,
        ComplexF64,
    )
    rng = StableRNG(1234)
    i = Index([U1(0) => 2, U1(1) => 3]; tags = "i")
    j = Index([U1(0) => 1, U1(1) => 2]; tags = "j")
    k = Index([U1(-1) => 1, U1(0) => 2]; tags = "k")

    a = randn(rng, elt, (i, j), (k,))
    @test isdual(inds(a)[3])
    m = matricize(a, (i, j), (k,))
    @test m isa AbstractMatrix{elt}
    @test size(m) == (length(i) * length(j), length(k))
    rt = unmatricize(m, (i, j), (k,))
    @test names(rt) == names(a)
    @test isdual(inds(rt)[3])
    @test unnamed(rt) ≈ unnamed(a)
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

# A dimension given as an index asserts its whole space, not just its length: over graded
# indices that means the sectors and the duality, both of which a length-only check would miss.
@testset "GradedArraysExt constructor space check" begin
    rng = StableRNG(1234)
    i = Index([U1(0) => 1, U1(1) => 2]; tags = "i")
    j = Index([U1(0) => 2, U1(1) => 1]; tags = "j")
    a = unnamed(randn(rng, (i, j)))
    @test ITensor(a, (i, j)) isa ITensor
    # Same length, opposite duality.
    @test length(dual(j)) == length(j)
    @test_throws ArgumentError ITensor(a, (i, dual(j)))
    # Same length, different sectors.
    k = Index([U1(0) => 1, U1(2) => 2]; tags = "k")
    @test length(k) == length(i)
    @test_throws ArgumentError ITensor(a, (k, j))
    # The codomain/domain form takes the domain index codomain-facing while the storage holds
    # it dualized, so the domain index is checked against the dual of the stored axis.
    m = unnamed(randn(rng, (i,), (j,)))
    @test ITensor(m, (i,), (j,)) isa ITensor
    @test_throws ArgumentError ITensor(m, (i,), (dual(j),))
end
