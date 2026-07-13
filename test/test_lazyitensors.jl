using AbstractTrees: AbstractTrees, print_tree, printnode
using Base.Broadcast: materialize
using ITensorBase: @names, Greedy, LazyNamedTensor, Mul, NamedTensor, SymbolicNamedTensor,
    dimnames, inds, ismul, lazy, nameddims, namedoneto, optimize_evaluation_order,
    substitute, symnameddims
using OMEinsumContractionOrders: ExhaustiveSearch, GreedyMethod, TreeSA
using TermInterface: arguments, arity, children, head, iscall, isexpr, maketerm, operation,
    sorted_arguments, sorted_children
using Test: @test, @test_broken, @test_throws, @testset
using WrappedUnions: unwrap

@testset "LazyNamedTensors" begin
    @testset "Basics" begin
        i, j, k, l = namedoneto.(2, (:i, :j, :k, :l))
        a1 = randn(i, j)
        a2 = randn(j, k)
        a3 = randn(k, l)
        l1, l2, l3 = lazy.((a1, a2, a3))
        for li in (l1, l2, l3)
            @test li isa LazyNamedTensor
            @test unwrap(li) isa NamedTensor
            @test inds(li) == inds(unwrap(li))
            @test copy(li) == unwrap(li)
            @test materialize(li) == unwrap(li)
        end
        l = l1 * l2 * l3
        @test copy(l) ≈ a1 * a2 * a3
        @test materialize(l) ≈ a1 * a2 * a3
        @test issetequal(inds(l), symdiff(inds.((a1, a2, a3))...))
        @test unwrap(l) isa Mul
        @test ismul(unwrap(l))
        @test unwrap(l).arguments == [l1 * l2, l3]
        # TermInterface.jl
        @test operation(unwrap(l)) ≡ *
        @test arguments(unwrap(l)) == [l1 * l2, l3]
    end

    @testset "TermInterface" begin
        a1 = nameddims(randn(2, 2), (:i, :j))
        a2 = nameddims(randn(2, 2), (:j, :k))
        a3 = nameddims(randn(2, 2), (:k, :l))
        l1, l2, l3 = lazy.((a1, a2, a3))

        @test_throws ErrorException arguments(l1)
        @test_throws ErrorException arity(l1)
        @test_throws ErrorException children(l1)
        @test_throws ErrorException head(l1)
        @test !iscall(l1)
        @test !isexpr(l1)
        @test_throws ErrorException operation(l1)
        @test_throws ErrorException sorted_arguments(l1)
        @test_throws ErrorException sorted_children(l1)
        @test AbstractTrees.children(l1) ≡ ()
        @test AbstractTrees.nodevalue(l1) ≡ a1
        @test sprint(show, l1) == sprint(show, a1)
        # The leaf format mirrors ITensorBase's display of a tensor's index names.
        @test sprint(printnode, l1) == "{\"i\", \"j\"}"
        @test sprint(print_tree, l1) == "{\"i\", \"j\"}\n"

        l = l1 * l2 * l3
        @test arguments(l) == [l1 * l2, l3]
        @test arity(l) == 2
        @test children(l) == [l1 * l2, l3]
        @test head(l) ≡ *
        @test iscall(l)
        @test isexpr(l)
        @test l == maketerm(LazyNamedTensor, *, [l1 * l2, l3], nothing)
        @test operation(l) ≡ *
        @test sorted_arguments(l) == [l1 * l2, l3]
        @test sorted_children(l) == [l1 * l2, l3]
        @test AbstractTrees.children(l) == [l1 * l2, l3]
        @test AbstractTrees.nodevalue(l) ≡ *
        @test sprint(show, l) == "(({\"i\", \"j\"} * {\"j\", \"k\"}) * {\"k\", \"l\"})"
        @test sprint(printnode, l) == "(({\"i\", \"j\"} * {\"j\", \"k\"}) * {\"k\", \"l\"})"
        @test sprint(print_tree, l) ==
            "(({\"i\", \"j\"} * {\"j\", \"k\"}) * {\"k\", \"l\"})\n" *
            "├─ ({\"i\", \"j\"} * {\"j\", \"k\"})\n" *
            "│  ├─ {\"i\", \"j\"}\n│  └─ {\"j\", \"k\"}\n" *
            "└─ {\"k\", \"l\"}\n"
    end

    @testset "symnameddims" begin
        a1, a2, a3 = symnameddims.((:a1, :a2, :a3))
        @test a1 isa LazyNamedTensor
        @test unwrap(a1) isa SymbolicNamedTensor
        @test unwrap(a1) == SymbolicNamedTensor(:a1, ())
        @test isequal(unwrap(a1), SymbolicNamedTensor(:a1, ()))
        @test isempty(inds(a1))
        @test isempty(dimnames(a1))

        ex = a1 * a2 * a3
        @test copy(ex) == ex
        @test arguments(ex) == [a1 * a2, a3]
        @test operation(ex) ≡ *
        @test sprint(show, ex) == "((a1 * a2) * a3)"
    end

    @testset "substitute" begin
        s = symnameddims.((:a1, :a2, :a3))
        i = @names i[1:4]
        a = (randn(2, 2)[i[1], i[2]], randn(2, 2)[i[2], i[3]], randn(2, 2)[i[3], i[4]])
        l = lazy.(a)

        seq = s[1] * (s[2] * s[3])
        net = substitute(seq, s .=> l)
        @test net == l[1] * (l[2] * l[3])
        @test arguments(net) == [l[1], l[2] * l[3]]
    end

    @testset "optimize_evaluation_order ($alg)" for alg in (Greedy(),)
        i, j, k, l = namedoneto.((2, 3, 4, 5), (:i, :j, :k, :l))
        s = [symnameddims(:a, (i, j)), symnameddims(:b, (j, k)), symnameddims(:c, (k, l))]
        flat = lazy(Mul(s))
        ordered = optimize_evaluation_order(flat; alg)
        @test ordered isa LazyNamedTensor
        @test ismul(ordered)
        # Reordering nests the flat product into binary contractions and preserves
        # the open indices.
        @test arity(ordered) == 2
        @test issetequal(dimnames(ordered), dimnames(flat))
    end

    @testset "optimize_evaluation_order (OMEinsumContractionOrders $alg)" for alg in
        (
            ExhaustiveSearch(),
            GreedyMethod(),
            TreeSA(),
        )
        i, j, k, l = namedoneto.((2, 3, 4, 5), (:i, :j, :k, :l))
        s = [symnameddims(:a, (i, j)), symnameddims(:b, (j, k)), symnameddims(:c, (k, l))]
        flat = lazy(Mul(s))
        ordered = optimize_evaluation_order(flat; alg)
        @test ordered isa LazyNamedTensor
        @test ismul(ordered)
        @test arity(ordered) == 2
        @test issetequal(dimnames(ordered), dimnames(flat))
    end
end
