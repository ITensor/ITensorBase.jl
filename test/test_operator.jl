using ITensorBase: ITensorBase as NDA, NamedTensor, NamedTensorOperator, apply, dimnames,
    id, inputname, inputnames, nameddims, namedoneto, operator, outputname, outputnames,
    product, replacedimnames, similar_operator, state, unname, unnamed
using LinearAlgebra: I, norm
using MatrixAlgebraKit: project_hermitian
using Random: Random
using StableRNGs: StableRNG
using TensorAlgebra.MatrixAlgebra:
    gram_eigh_full, gram_eigh_full_with_pinv, invsqrth_safe, sqrth_invsqrth_safe, sqrth_safe
using TensorAlgebra: matricize
using Test: @test, @test_throws, @testset

@testset "operator" begin
    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    @test o isa NamedTensorOperator{String}
    @test eltype(o) ≡ Float64
    @test issetequal(NDA.outputnames(o), ("i'", "j'"))
    @test issetequal(NDA.inputnames(o), ("i", "j"))

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    õ = similar(o)
    @test õ isa NamedTensorOperator{String}
    @test eltype(õ) ≡ Float64
    @test issetequal(NDA.outputnames(õ), ("i'", "j'"))
    @test issetequal(NDA.inputnames(õ), ("i", "j"))

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    õ = similar(o, Float32)
    @test õ isa NamedTensorOperator{String}
    @test eltype(õ) ≡ Float32
    @test issetequal(NDA.outputnames(õ), ("i'", "j'"))
    @test issetequal(NDA.inputnames(õ), ("i", "j"))

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

@testset "wire lookups" begin
    o = operator(randn(2, 2), ("i'",), ("i",))
    # Three-argument form returns the default for an unpaired name; two-argument form throws.
    @test outputname(o, "i", "x") == "i'"
    @test inputname(o, "i'", "x") == "i"
    @test outputname(o, "z", "x") == "x"
    @test outputname(o, "i") == "i'"
    @test inputname(o, "i'") == "i"
    @test_throws ArgumentError outputname(o, "z")
    # A plain tensor has no pairing, so every two-argument lookup throws.
    @test_throws ArgumentError inputname(NamedTensor(randn(2), ("i",)), "i")
end

@testset "apply composition" begin
    # Operator applied to another operator: `x` lands on `y`'s output, and the result is an
    # operator on `x`'s input space keeping `y`'s input.
    A = operator(randn(2, 2), ("k",), ("j",))
    B = operator(randn(2, 2), ("j",), ("i",))
    AB = apply(A, B)
    @test AB isa NamedTensorOperator
    @test issetequal(outputnames(AB), ("j",))
    @test issetequal(inputnames(AB), ("i",))
    Am = unname(state(A), ("k", "j"))
    Bm = unname(state(B), ("j", "i"))
    @test unname(state(AB), ("j", "i")) ≈ Am * Bm

    # Applying to a bare state gives a bare state.
    v = NamedTensor(randn(2, 2), ("j", "i"))
    Av = apply(A, v)
    @test Av isa NamedTensor
    @test issetequal(dimnames(Av), ("j", "i"))
    @test unname(Av, ("j", "i")) ≈ Am * unname(v, ("j", "i"))

    # Disjoint apply tensors, like `*`/product.
    C = operator(randn(2, 2), ("m",), ("n",))
    D = operator(randn(2, 2), ("j",), ("i",))
    CD = apply(C, D)
    @test CD isa NamedTensorOperator
    @test issetequal(outputnames(CD), ("m", "j"))
    @test issetequal(inputnames(CD), ("n", "i"))

    # Applying into `y`'s input side (not its codomain) errors.
    @test_throws ArgumentError apply(
        operator(randn(2, 2), ("k",), ("j",)), operator(randn(2, 2), ("i",), ("j",))
    )
end

@testset "product composition" begin
    # Same-space single site: product is matrix multiplication, and it is
    # bidirectional (product(B, A) is the other matrix order).
    A = operator(randn(2, 2), ("a'",), ("a",))
    B = operator(randn(2, 2), ("a'",), ("a",))
    Am = unname(state(A), ("a'", "a"))
    Bm = unname(state(B), ("a'", "a"))

    AB = product(A, B)
    @test AB isa NamedTensorOperator
    @test issetequal(outputnames(AB), ("a'",))
    @test issetequal(inputnames(AB), ("a",))
    @test unname(state(AB), ("a'", "a")) ≈ Am * Bm

    BA = product(B, A)
    @test unname(state(BA), ("a'", "a")) ≈ Bm * Am

    # Disjoint operators tensor into a two-site operator.
    C = operator(randn(2, 2), ("i'",), ("i",))
    D = operator(randn(3, 3), ("j'",), ("j",))
    CD = product(C, D)
    @test issetequal(outputnames(CD), ("i'", "j'"))
    @test issetequal(inputnames(CD), ("i", "j"))

    # Partial overlap: A on sites (1, 2), B on sites (2, 3), sharing site 2 →
    # a three-site operator composed on 2 and tensored on 1 and 3.
    A3 = operator(
        nameddims(randn(2, 2, 2, 2), ("1'", "2'", "1", "2")),
        ("1'", "2'"),
        ("1", "2")
    )
    B3 = operator(
        nameddims(randn(2, 2, 2, 2), ("2'", "3'", "2", "3")),
        ("2'", "3'"),
        ("2", "3")
    )
    AB3 = product(A3, B3)
    @test issetequal(outputnames(AB3), ("1'", "2'", "3'"))
    @test issetequal(inputnames(AB3), ("1", "2", "3"))
    # Value: weld A's input 2 to B's output 2' and contract.
    manual =
        replacedimnames(state(A3), "2" => "bond") *
        replacedimnames(state(B3), "2'" => "bond")
    order = ("1'", "2'", "3'", "1", "2", "3")
    @test unname(state(AB3), order) ≈ unname(manual, order)

    # Batched: an operator applied to a state with a spectator index. The shared `i`
    # contracts (a's input meets b's leg), `j` rides along, and the output stays on `i'`.
    Aop = operator(randn(2, 2), ("i'",), ("i",))
    v = NamedTensor(randn(2, 3), ("i", "j"))
    Av = product(Aop, v)
    @test issetequal(dimnames(Av), ("i'", "j"))
    @test isempty(outputnames(Av))
    @test isempty(inputnames(Av))
    @test unname(state(Av), ("i'", "j")) ≈
        unname(state(Aop), ("i'", "i")) * unname(v, ("i", "j"))

    # Kraus: composing two operators that share both the i-wire and a dangling Kraus
    # index `j` welds the wire and sums over `j`.
    K = operator(nameddims(randn(2, 2, 3), ("i'", "i", "j")), ("i'",), ("i",))
    KK = product(K, K)
    @test issetequal(outputnames(KK), ("i'",))
    @test issetequal(inputnames(KK), ("i",))
    @test issetequal(dimnames(KK), ("i'", "i"))  # j contracted away
    Km = unname(state(K), ("i'", "i", "j"))
    manual_kraus = sum(Km[:, :, jj] * Km[:, :, jj] for jj in axes(Km, 3))
    @test unname(state(KK), ("i'", "i")) ≈ manual_kraus

    # Overlapping but mismatched wires are rejected. Shared input, differing output:
    @test_throws ArgumentError product(
        operator(randn(2, 2), ("j",), ("i",)), operator(randn(2, 2), ("k",), ("i",))
    )
    # ...and, symmetrically, shared output, differing input:
    @test_throws ArgumentError product(
        operator(randn(2, 2), ("j",), ("i",)), operator(randn(2, 2), ("j",), ("k",))
    )

    # Chaining (`a`'s output is `b`'s input) is not composition on a shared site; it is
    # end-to-end contraction, so `product` rejects it and `*` handles it, returning the
    # composed operator `a'←b`.
    chain_a = operator(randn(2, 2), ("a'",), ("m",))
    chain_b = operator(randn(2, 2), ("m",), ("b",))
    @test_throws ArgumentError product(chain_a, chain_b)
    chain = chain_a * chain_b
    @test issetequal(outputnames(chain), ("a'",))
    @test issetequal(inputnames(chain), ("b",))
end

@testset "operator from named ranges" begin
    # Output/input may be given as named ranges, not just names.
    i, ip = namedoneto(2, "i"), namedoneto(2, "i'")
    o = operator(randn(2, 2), [ip], [i])
    @test o isa NamedTensorOperator{String}
    @test issetequal(outputnames(o), ("i'",))
    @test issetequal(inputnames(o), ("i",))
end

@testset "one(::NamedTensorOperator)" begin
    # Identity-operator construction: matricized form is the identity matrix.
    i, j, k, l = namedoneto.((2, 3, 2, 3), ("i", "j", "k", "l"))
    op = operator(randn(i, j, k, l), ("i", "j"), ("k", "l"))
    Id = one(op)
    @test Id isa NamedTensorOperator{String}
    @test outputnames(Id) == outputnames(op)
    @test inputnames(Id) == inputnames(op)
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

@testset "id(elt, codomain, domain)" begin
    # From-scratch identity map (no prototype): matricized form is the identity matrix.
    i, j, k, l = namedoneto.((2, 3, 2, 3), ("i", "j", "k", "l"))
    Id = id(Float64, (i, j), (k, l))
    @test eltype(Id) === Float64
    @test issetequal(dimnames(Id), ("i", "j", "k", "l"))
    Id_mat = matricize(Id, (i, j) => "row", (k, l) => "col")
    @test unname(Id_mat, ("row", "col")) ≈ I(6)

    # The requested element type is honored.
    @test eltype(id(ComplexF64, (i, j), (k, l))) === ComplexF64
end

@testset "similar_operator" begin
    # Five-arg canonical: explicit element type, axes, output, input names.
    op = similar_operator(randn(3, 3), Float32, (Base.OneTo(3),), ("i'",), ("i",))
    @test op isa NamedTensorOperator{String}
    @test issetequal(outputnames(op), ("i'",))
    @test issetequal(inputnames(op), ("i",))

    # Output names default to fresh `uniquename` outputs.
    op = similar_operator(randn(3, 3), Float64, (Base.OneTo(3),), ("i",))
    @test op isa NamedTensorOperator{String}
    @test issetequal(inputnames(op), ("i",))
    @test only(outputnames(op)) != "i"

    # Named-axes form reuses each axis's name as the input.
    i = namedoneto(3, "i")
    op = similar_operator(randn(3, 3), Float64, (i,))
    @test issetequal(inputnames(op), ("i",))
    @test only(outputnames(op)) != "i"

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
    # preserve the `NamedTensorOperator` wrapper and its output/input pairing.
    # (Contraction `*` is operator-preserving too, in its own testset below.)
    o = operator(randn(2, 2), ("i'",), ("i",))
    s = state(o)
    nms = ("i'", "i")

    for r in (o + o, o - o, -o, 2 * o, o * 2, 2 .* o, o .* 2, o ./ 2)
        @test r isa NamedTensorOperator
        @test issetequal(outputnames(r), ("i'",))
        @test issetequal(inputnames(r), ("i",))
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
    # scalar with no surviving output/input. It is still an `NamedTensorOperator`
    # (with empty output/input), so the product type does not depend on which
    # names happen to contract.
    oo = o * o
    @test oo isa NamedTensorOperator
    @test isempty(outputnames(oo))
    @test isempty(inputnames(oo))

    # Operator combined with a non-operator tensor is rejected.
    plain = NamedTensor(randn(2, 2), ("i'", "i"))
    @test_throws ArgumentError o .+ plain

    # Two operators whose name sets match but whose output/input split differs
    # are rejected (the split would otherwise be ambiguous).
    o_swapped = operator(randn(2, 2), ("i",), ("i'",))
    @test_throws ArgumentError o .+ o_swapped
end

@testset "operator-preserving contraction" begin
    # A shared *dangling* leg (in neither pairing) is summed away, and the
    # surviving output/input of each operand combine. This is the `c† * c`
    # hopping pattern: two operators paired over an auxiliary link.
    a = operator(nameddims(randn(2, 2, 3), ("i'", "i", "aux")), ["i'"], ["i"])
    b = operator(nameddims(randn(2, 2, 3), ("j'", "j", "aux")), ["j'"], ["j"])
    ab = a * b
    @test ab isa NamedTensorOperator
    @test issetequal(outputnames(ab), ("i'", "j'"))
    @test issetequal(inputnames(ab), ("i", "j"))
    @test !("aux" in dimnames(ab))
    @test state(ab) ≈ state(a) * state(b)

    # A shared *paired* index (a's input equals b's output) chains through the
    # contraction: a's output partner pairs with b's input partner.
    A = operator(randn(2, 2), ("a'",), ("m",))
    B = operator(randn(2, 2), ("m",), ("b",))
    AB = A * B
    @test AB isa NamedTensorOperator
    @test issetequal(outputnames(AB), ("a'",))
    @test issetequal(inputnames(AB), ("b",))

    # Applying an operator to a plain state contracts the operator's input and
    # leaves its output dangling. The result stays an `NamedTensorOperator` with empty
    # output/input (the surviving `a'` leg is dangling, in neither).
    v = nameddims(randn(2), ("m",))
    Av = operator(randn(2, 2), ("a'",), ("m",)) * v
    @test Av isa NamedTensorOperator
    @test isempty(outputnames(Av))
    @test isempty(inputnames(Av))
    @test issetequal(dimnames(Av), ("a'",))
end

@testset "gram_eigh_full on NamedTensorOperator" begin
    n = 5
    B = randn(n, n)
    A = B * B'  # Hermitian PSD
    M_op = operator(A, ["ket"], ["bra"])

    X_op = gram_eigh_full(M_op)
    X_arr = gram_eigh_full(nameddims(A, ("ket", "bra")), ("ket",), ("bra",))
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

@testset "Hermitian square roots on NamedTensorOperator" begin
    n = 5
    B = randn(n, n)
    A = B * B'  # Hermitian PSD
    M_op = operator(A, ["ket"], ["bra"])

    # `project_hermitian` keeps the operator structure; a non-Hermitian input maps to its
    # Hermitian part.
    H_op = project_hermitian(M_op)
    @test H_op isa NamedTensorOperator
    @test outputnames(H_op) == outputnames(M_op)
    @test inputnames(H_op) == inputnames(M_op)
    @test H_op ≈ M_op
    @test project_hermitian(operator(B, ["ket"], ["bra"])) ≈
        operator((B + B') / 2, ["ket"], ["bra"])

    # The roots are again bond operators, with the same codomain/domain as the input.
    for X in (sqrth_safe(M_op), invsqrth_safe(M_op), sqrth_invsqrth_safe(M_op)...)
        @test X isa NamedTensorOperator
        @test outputnames(X) == outputnames(M_op)
        @test inputnames(X) == inputnames(M_op)
    end

    P = unnamed(state(sqrth_safe(M_op)))
    @test P * P' ≈ A
    @test unnamed(state(invsqrth_safe(M_op))) * P ≈ I(n)

    Psqrt, Pinv = sqrth_invsqrth_safe(M_op)
    Pmat = unnamed(state(Psqrt))
    @test Pmat * Pmat' ≈ A
    @test Pmat * unnamed(state(Pinv)) ≈ I(n)
end
