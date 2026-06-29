using ITensorBase: ITensorBase as NDA, NamedTensor, NamedTensorOperator, apply,
    codomainnames, dimnames, domainnames, nameddims, namedoneto, operator, product,
    replacedimnames, similar_operator, state, unname, unnamed
using LinearAlgebra: I, norm
using Random: Random
using StableRNGs: StableRNG
using TensorAlgebra.MatrixAlgebra: gram_eigh_full, gram_eigh_full_with_pinv
using TensorAlgebra: matricize
using Test: @test, @test_throws, @testset

@testset "operator" begin
    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    @test o isa NamedTensorOperator{String}
    @test eltype(o) ≡ Float64
    @test issetequal(NDA.codomainnames(o), ("i'", "j'"))
    @test issetequal(NDA.domainnames(o), ("i", "j"))

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    õ = similar(o)
    @test õ isa NamedTensorOperator{String}
    @test eltype(õ) ≡ Float64
    @test issetequal(NDA.codomainnames(õ), ("i'", "j'"))
    @test issetequal(NDA.domainnames(õ), ("i", "j"))

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    õ = similar(o, Float32)
    @test õ isa NamedTensorOperator{String}
    @test eltype(õ) ≡ Float32
    @test issetequal(NDA.codomainnames(õ), ("i'", "j'"))
    @test issetequal(NDA.domainnames(õ), ("i", "j"))

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    @test o isa NamedTensorOperator
    o² = product(o, o)
    @test issetequal(dimnames(o²), ("i'", "j'", "i", "j"))
    õ = replacedimnames(
        state(o), "i" => "i'", "j" => "j'", "i'" => "x", "j'" => "y"
    )
    o²′ = replacedimnames(õ * o, "x" => "i'", "y" => "j'")
    @test state(o²) ≈ o²′

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    v = NamedTensor(randn(2, 2), ("i", "j"))
    ov = apply(o, v)
    @test issetequal(dimnames(ov), ("i", "j"))
    @test ov ≈ replacedimnames(o * v, "i'" => "i", "j'" => "j")
end

@testset "operator from named ranges" begin
    # Codomain/domain may be given as named ranges, not just names.
    i, ip = namedoneto(2, "i"), namedoneto(2, "i'")
    o = operator(randn(2, 2), [ip], [i])
    @test o isa NamedTensorOperator{String}
    @test issetequal(codomainnames(o), ("i'",))
    @test issetequal(domainnames(o), ("i",))
end

@testset "one(::NamedTensorOperator)" begin
    # Identity-operator construction: matricized form is the identity matrix.
    i, j, k, l = namedoneto.((2, 3, 2, 3), ("i", "j", "k", "l"))
    op = operator(randn(i, j, k, l), ("i", "j"), ("k", "l"))
    Id = one(op)
    @test Id isa NamedTensorOperator{String}
    @test codomainnames(Id) == codomainnames(op)
    @test domainnames(Id) == domainnames(op)
    Id_mat = matricize(state(Id), (i, j) => "row", (k, l) => "col")
    @test unname(Id_mat, ("row", "col")) ≈ I(6)
end

@testset "one(::AbstractNamedTensor, codomain, domain)" begin
    # Trivial codomain/domain layout.
    i, j, k, l = namedoneto.((2, 3, 2, 3), ("i", "j", "k", "l"))
    a = randn(i, j, k, l)
    Id = one(a, (i, j), (k, l))
    Id_mat = matricize(Id, (i, j) => "row", (k, l) => "col")
    @test unname(Id_mat, ("row", "col")) ≈ I(6)

    # Non-trivial axis ordering: codomain/domain are interleaved in `a`.
    p, q, r, s = namedoneto.((2, 4, 2, 4), ("p", "q", "r", "s"))
    a = randn(p, r, q, s)  # storage order interleaves codomain (p, q) and domain (r, s)
    Id = one(a, (p, q), (r, s))
    @test issetequal(dimnames(Id), ("p", "r", "q", "s"))
    Id_mat = matricize(Id, (p, q) => "row", (r, s) => "col")
    @test unname(Id_mat, ("row", "col")) ≈ I(8)
end

@testset "similar_operator" begin
    # Five-arg canonical: explicit element type, axes, codomain, domain names.
    op = similar_operator(randn(3, 3), Float32, (Base.OneTo(3),), ("i'",), ("i",))
    @test op isa NamedTensorOperator{String}
    @test issetequal(codomainnames(op), ("i'",))
    @test issetequal(domainnames(op), ("i",))

    # Codomain names default to fresh `uniquename` outputs.
    op = similar_operator(randn(3, 3), Float64, (Base.OneTo(3),), ("i",))
    @test op isa NamedTensorOperator{String}
    @test issetequal(domainnames(op), ("i",))
    @test only(codomainnames(op)) != "i"

    # Named-axes form reuses each axis's name as the domain.
    i = namedoneto(3, "i")
    op = similar_operator(randn(3, 3), Float64, (i,))
    @test issetequal(domainnames(op), ("i",))
    @test only(codomainnames(op)) != "i"

    # Element type defaults to `eltype(prototype)`.
    op = similar_operator(randn(ComplexF32, 3, 3), (Base.OneTo(3),), ("i'",), ("i",))
    @test eltype(op) === ComplexF32
end

@testset "randn!(::NamedTensorOperator) / rand!" begin
    op = operator(zeros(3, 3), ("i'",), ("i",))
    rng = StableRNG(123)
    Random.randn!(rng, op)
    @test !all(iszero, unnamed(state(op)))

    Random.rand!(rng, op)
    @test !all(iszero, unnamed(state(op)))
    @test all(0 .≤ unnamed(state(op)) .≤ 1)
end

@testset "operator-preserving broadcasting" begin
    # `+`, `-`, and scalar multiplication lower to broadcasting. An operator
    # broadcasts as itself (it is not peeled to its `state`), so these operations
    # preserve the `NamedTensorOperator` wrapper and its codomain/domain bijection.
    # (Contraction `*` is operator-preserving too, in its own testset below.)
    o = operator(randn(2, 2), ("i'",), ("i",))
    s = state(o)
    nms = ("i'", "i")

    for r in (o + o, o - o, -o, 2 * o, o * 2, 2 .* o, o .* 2, o ./ 2)
        @test r isa NamedTensorOperator
        @test issetequal(codomainnames(r), ("i'",))
        @test issetequal(domainnames(r), ("i",))
    end

    @test unname(state(o + o), nms) ≈ 2 .* unname(s, nms)
    @test all(iszero, unname(state(o - o), nms))
    @test unname(state(-o), nms) ≈ -unname(s, nms)
    @test unname(state(2 * o), nms) ≈ 2 .* unname(s, nms)
    @test unname(state(o * 2), nms) ≈ 2 .* unname(s, nms)
    @test unname(state(2 .* o), nms) ≈ 2 .* unname(s, nms)
    @test unname(state(o .* 2), nms) ≈ 2 .* unname(s, nms)
    @test unname(state(o ./ 2), nms) ≈ unname(s, nms) ./ 2

    # `o` shares both its names with itself, so `o * o` fully contracts to a
    # scalar with no surviving codomain/domain. It is still an `NamedTensorOperator`
    # (with empty codomain/domain), so the product type does not depend on which
    # names happen to contract.
    oo = o * o
    @test oo isa NamedTensorOperator
    @test isempty(codomainnames(oo))
    @test isempty(domainnames(oo))

    # Operator combined with a non-operator tensor is rejected.
    plain = NamedTensor(randn(2, 2), ("i'", "i"))
    @test_throws ArgumentError o .+ plain

    # Two operators whose name sets match but whose codomain/domain split differs
    # are rejected (the split would otherwise be ambiguous).
    o_swapped = operator(randn(2, 2), ("i",), ("i'",))
    @test_throws ArgumentError o .+ o_swapped
end

@testset "operator-preserving contraction" begin
    # A shared *dangling* leg (in neither bijection) is summed away, and the
    # surviving codomain/domain of each operand combine. This is the `c† * c`
    # hopping pattern: two operators paired over an auxiliary link.
    a = operator(nameddims(randn(2, 2, 3), ("i'", "i", "aux")), ["i'"], ["i"])
    b = operator(nameddims(randn(2, 2, 3), ("j'", "j", "aux")), ["j'"], ["j"])
    ab = a * b
    @test ab isa NamedTensorOperator
    @test issetequal(codomainnames(ab), ("i'", "j'"))
    @test issetequal(domainnames(ab), ("i", "j"))
    @test !("aux" in dimnames(ab))
    @test state(ab) ≈ state(a) * state(b)

    # A shared *paired* index (a's domain equals b's codomain) chains through the
    # contraction: a's codomain partner pairs with b's domain partner.
    A = operator(randn(2, 2), ("a'",), ("m",))
    B = operator(randn(2, 2), ("m",), ("b",))
    AB = A * B
    @test AB isa NamedTensorOperator
    @test issetequal(codomainnames(AB), ("a'",))
    @test issetequal(domainnames(AB), ("b",))

    # Applying an operator to a plain state contracts the operator's domain and
    # leaves its output dangling. The result stays an `NamedTensorOperator` with empty
    # codomain/domain (the surviving `a'` leg is dangling, in neither).
    v = nameddims(randn(2), ("m",))
    Av = operator(randn(2, 2), ("a'",), ("m",)) * v
    @test Av isa NamedTensorOperator
    @test isempty(codomainnames(Av))
    @test isempty(domainnames(Av))
    @test issetequal(dimnames(Av), ("a'",))
end

@testset "gram_eigh_full on NamedTensorOperator" begin
    n = 5
    B = randn(n, n)
    A = B * B'  # Hermitian PSD
    M_nda = nameddims(A, ("ket", "bra"))
    M_op = operator(M_nda, ["ket"], ["bra"])

    X_op = gram_eigh_full(M_op)
    X_arr = gram_eigh_full(M_nda, ("ket",), ("bra",))
    # Operator entry forwards to the named-array entry: same data, same shape.
    @test size(parent(X_op)) == size(parent(X_arr))

    Xp = parent(X_op)
    @test Xp * Xp' ≈ A

    X2, Y2 = gram_eigh_full_with_pinv(M_op)
    Xp2 = parent(X2)
    Yp2 = parent(Y2)
    @test Xp2 * Xp2' ≈ A
    @test Yp2 * Xp2 ≈ I(n)
end
