using Combinatorics: Combinatorics
using ITensorBase: @names, AbstractNamedTensor, Name, NameMismatch, NamedDimsCartesianIndex,
    NamedDimsCartesianIndices, NamedTensor, aligndims, aligneddims, apply, dim, dimnames,
    dimnametype, dims, inds, isnamed, mapinds, name, named, nameddims, namedoneto, product,
    replacedimnames, replaceinds, setdimnames, unname, unnamed, unnamedtype
using LinearAlgebra: LinearAlgebra
using TensorAlgebra: datatype
using Test: @test, @test_throws, @testset
using VectorInterface: scalartype

module TestBasicsUtils
    elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
end

@testset "ITensorBase.jl" begin
    @testset "Basic functionality (eltype=$elt)" for elt in TestBasicsUtils.elts
        a = randn(elt, 3, 4)
        @test !isnamed(a)
        na = nameddims(a, ("i", "j"))
        @test na isa NamedTensor{String}
        @test na isa AbstractNamedTensor{String}
        @test eltype(na) === elt
        @test ndims(na) == 2
        @test_throws MethodError unnamed(a)
        @test_throws MethodError unname(a, ("i", "j"))
        @test_throws MethodError unnamed(a, ("i", "j"))
        @test unnamed(na) == a
        si, sj = size(na)
        ai, aj = axes(na)
        i = namedoneto(3, "i")
        j = namedoneto(4, "j")
        @test si == 3
        @test sj == 4
        @test name(ai) == "i"
        @test name(aj) == "j"
        @test isnamed(na)
        @test inds(na) == [i, j]
        @test inds(na) isa Vector
        @test axes(na) == (i, j)
        @test axes(na) isa Tuple
        @test inds(na, 1) == i
        @test inds(na, 2) == j
        @test dimnames(na) == ["i", "j"]
        @test dimnames(na, 1) == "i"
        @test dimnames(na, 2) == "j"
        @test dim(na, "i") == 1
        @test dim(na, "j") == 2
        @test dims(na, ("j", "i")) == (2, 1)
        @test na[1, 1] == a[1, 1]
        # The parent array's concrete type is erased from the type but is still
        # recoverable from an instance.
        @test unnamedtype(na) === typeof(a)
        @test unnamedtype(typeof(na)) === AbstractArray
        @test dimnametype(typeof(na)) === String
        @test dimnametype(na) === String

        # equals (==)/isequal
        a = randn(elt, 3, 4)
        na = nameddims(a, ("i", "j"))
        @test na == na
        @test na == aligndims(na, ("j", "i"))
        @test isequal(na, na)
        @test isequal(na, aligndims(na, ("j", "i")))
        @test hash(na) == hash(aligndims(na, ("j", "i")))
        # Regression test that ITensorBase
        # with different names are not equal (as opposed to
        # erroring).
        @test na ≠ nameddims(a, ("j", "k"))
        @test !isequal(na, nameddims(a, ("j", "k")))
        @test hash(na) ≠ hash(nameddims(a, ("j", "k")))

        a = randn(elt, 2, 2)
        na = nameddims(a, ("i", "j"))
        @test CartesianIndices(na) == CartesianIndices(a)
        @test collect(pairs(na)) == (CartesianIndices(a) .=> a)

        @test_throws ArgumentError NamedTensor(
            randn(4),
            namedoneto.((2, 2), ("i", "j"))
        )
        ## @test_throws ErrorException NamedTensor(randn(2, 2), namedoneto.((2, 3), ("i", "j")))

        a = randn(elt, 3, 4)
        na = nameddims(a, ("i", "j"))
        @test eltype(na) ≡ elt
        @test scalartype(na) ≡ elt
        @test datatype(na) ≡ typeof(a)
        a′ = Array(na)
        @test eltype(a′) ≡ elt
        @test a′ isa Matrix{elt}
        @test a′ == a

        if elt <: Real
            a = randn(elt, 3, 4)
            na = nameddims(a, ("i", "j"))
            for a′ in (Array{Float32}(na), Matrix{Float32}(na))
                @test eltype(a′) ≡ Float32
                @test a′ isa Matrix{Float32}
                @test a′ == Float32.(a)
            end
        end

        a = randn(elt, 2, 2, 2)
        na = nameddims(a, ("i", "j", "k"))
        b = randn(elt, 2, 2, 2)
        nb = nameddims(b, ("k", "i", "j"))
        copyto!(na, nb)
        @test na == nb
        @test unnamed(na) == unname(nb, ("i", "j", "k"))
        @test unnamed(na) == permutedims(unnamed(nb), (2, 3, 1))

        a = randn(elt, 3, 4)
        na = nameddims(a, ("i", "j"))
        i = namedoneto(3, "i")
        j = namedoneto(4, "j")
        for na′ in (
                similar(na, Float32, (j, i)),
                similar(a, Float32, (j, i)),
            )
            @test eltype(na′) ≡ Float32
            @test all(inds(na′) .== (j, i))
            @test na′ ≠ na
        end

        a = randn(elt, 3, 4)
        na = nameddims(a, ("i", "j"))
        i = namedoneto(3, "i")
        j = namedoneto(4, "j")
        for na′ in (
                similar(na, (j, i)),
                similar(a, (j, i)),
            )
            @test eltype(na′) ≡ eltype(na)
            @test all(inds(na′) .== (j, i))
            @test na′ ≠ na
        end

        # getindex syntax
        i = Name("i")
        j = Name("j")
        @test a[i, j] == na
        @test @view(a[i, j]) == na
        @test na[j[1], i[2]] == a[2, 1]
        @test inds(na[j, i]) == [named(1:3, "i"), named(1:4, "j")]
        @test na[j, i] == na
        @test @view(na[j, i]) == na
        @test i[axes(a, 1)] == named(1:3, "i")
        @test j[axes(a, 2)] == named(1:4, "j")
        @test axes(na, i) == ai
        @test axes(na, j) == aj
        @test size(na, i) == si
        @test size(na, j) == sj

        # Regression test for ambiguity error with
        # `Base.getindex(A::Array, I::AbstractUnitRange{<:Integer})`.
        i = namedoneto(2, "i")
        a = randn(elt, 2)
        na = a[i]
        @test na isa NamedTensor{String}
        @test dimnames(na) == ["i"]
        @test unnamed(na) == a

        # slicing
        a = randn(elt, 3, 3)
        na = NamedTensor(a, ("i", "j"))
        for na′ in (na[named(2:3, "i"), named(2:3, "j")], na["i" => 2:3, "j" => 2:3])
            @test inds(na′) == [named(1:2, "i"), named(1:2, "j")]
            @test unnamed(na′) == a[2:3, 2:3]
            @test unnamed(na′) isa typeof(a)
        end

        # view slicing
        a = randn(elt, 3, 3)
        na = NamedTensor(a, ("i", "j"))
        for na′ in
            (@view(na[named(2:3, "i"), named(2:3, "j")]), @view(na["i" => 2:3, "j" => 2:3]))
            @test inds(na′) == [named(1:2, "i"), named(1:2, "j")]
            @test copy(unnamed(na′)) == a[2:3, 2:3]
            @test unnamed(na′) ≡ @view(a[2:3, 2:3])
            @test unnamed(na′) isa SubArray{elt, 2}
        end

        # aliasing
        a′ = randn(elt, 2, 2)
        i = Name("i")
        j = Name("j")
        a′ij = @view a′[i, j]
        a′ij[i[1], j[2]] = 12
        @test a′ij[i[1], j[2]] == 12
        @test a′[1, 2] == 12
        a′ji = @view a′ij[j, i]
        a′ji[i[2], j[1]] = 21
        @test a′ji[i[2], j[1]] == 21
        @test a′ij[i[2], j[1]] == 21
        @test a′[2, 1] == 21

        a′ = randn(elt, 2, 2)
        i = Name("i")
        j = Name("j")
        a′ij = a′[i, j]
        a′ij[i[1], j[2]] = 12
        @test a′ij[i[1], j[2]] == 12
        @test a′[1, 2] ≠ 12
        a′ji = a′ij[j, i]
        a′ji[i[2], j[1]] = 21
        @test a′ji[i[2], j[1]] == 21
        @test a′ij[i[2], j[1]] ≠ 21
        @test a′[2, 1] ≠ 21

        a = randn(elt, 3, 4)
        na = nameddims(a, ("i", "j"))
        a′ = unnamed(na)
        @test a′ isa Matrix{elt}
        @test a′ == a
        a′ = unname(na, ("j", "i"))
        @test a′ isa Matrix{elt}
        @test a′ == transpose(a)
        a′ = unnamed(na, ("j", "i"))
        @test a′ isa PermutedDimsArray{elt}
        @test a′ == transpose(a)
        nb = setdimnames(na, ("k", "j"))
        @test inds(nb) == [named(1:3, "k"), named(1:4, "j")]
        @test unnamed(nb) == a
        nb = replacedimnames(na, "i" => "k")
        @test inds(nb) == [named(1:3, "k"), named(1:4, "j")]
        @test unnamed(nb) == a
        nb = replaceinds(na, named(1:3, "i") => named(1:3, "k"))
        @test inds(nb) == [named(1:3, "k"), named(1:4, "j")]
        @test unnamed(nb) == a
        nb = replaceinds(n -> n == named(1:3, "i") ? named(1:3, "k") : n, na)
        @test inds(nb) == [named(1:3, "k"), named(1:4, "j")]
        @test unnamed(nb) == a
        nb = mapinds(n -> n == named(1:3, "i") ? named(1:3, "k") : n, na)
        @test inds(nb) == [named(1:3, "k"), named(1:4, "j")]
        @test unnamed(nb) == a
        na[1, 1] = 11
        @test na[1, 1] == 11
        @test size(na) == (3, 4)
        # An NamedTensor's `length` is the plain element count (product of its size).
        @test length(na) == 12
        @test Tuple(axes(na)) == (named(1:3, "i"), named(1:4, "j"))
        @test randn(named.((3, 4), ("i", "j"))) isa NamedTensor
        @test na["i" => 1, "j" => 2] == a[1, 2]
        @test na["j" => 2, "i" => 1] == a[1, 2]
        na["j" => 2, "i" => 1] = 12
        @test na[1, 2] == 12
        @test na[j => 1, i => 2] == a[2, 1]
        na[j => 1, i => 2] = 21
        @test na[2, 1] == 21
        na′ = aligndims(na, ("j", "i"))
        @test unnamed(na′) isa Matrix{elt}
        @test a == permutedims(unnamed(na′), (2, 1))
        na′ = aligneddims(na, ("j", "i"))
        @test unnamed(na′) isa PermutedDimsArray{elt}
        @test a == permutedims(unnamed(na′), (2, 1))
        na′ = aligndims(na, (j, i))
        @test unnamed(na′) isa Matrix{elt}
        @test a == permutedims(unnamed(na′), (2, 1))
        na′ = aligneddims(na, (j, i))
        @test unnamed(na′) isa PermutedDimsArray{elt}
        @test a == permutedims(unnamed(na′), (2, 1))
        # The map form of `aligndims` takes a codomain and a domain tuple. A dense backend
        # ignores the split and stores the reordered result flat, matching the flat form.
        na′ = aligndims(na, (j,), (i,))
        @test unnamed(na′) isa Matrix{elt}
        @test a == permutedims(unnamed(na′), (2, 1))
        # Two-tuple `randn`/`zeros` take a codomain and a domain index tuple; a dense backend
        # ignores the split and stores flat, named by the codomain then the domain.
        ci, cj = namedoneto(3, "i"), namedoneto(4, "j")
        nab = randn(elt, (ci,), (cj,))
        @test unnamed(nab) isa Matrix{elt}
        @test dimnames(nab) == [name(ci), name(cj)]
        @test unnamed(zeros(elt, (ci,), (cj,))) == zeros(elt, 3, 4)
        # An empty codomain lands every index in the domain, the mirror of an empty domain. A
        # dense backend ignores the split, so these match the flat forms.
        na′ = aligndims(na, (), (j, i))
        @test unnamed(na′) isa Matrix{elt}
        @test a == permutedims(unnamed(na′), (2, 1))
        nbra = randn(elt, (), (cj,))
        @test unnamed(nbra) isa Vector{elt}
        @test dimnames(nbra) == [name(cj)]
        @test unnamed(zeros(elt, (), (cj,))) == zeros(elt, 4)
        # An all-empty split has no map meaning, so it errors rather than recursing or
        # silently building a scalar.
        @test_throws MethodError randn(elt, (), ())
        @test_throws MethodError zeros(elt, (), ())

        na = nameddims(randn(elt, 2, 3), (:i, :j))
        nb = nameddims(randn(elt, 3, 2), (:j, :i))
        nc = zeros(elt, named.((2, 3), (:i, :j)))
        Is = eachindex(na, nb)
        @test Is isa NamedDimsCartesianIndices{2}
        @test issetequal(Is.indices, (named(1:2, :i), named(1:3, :j)))
        for I in Is
            @test I isa NamedDimsCartesianIndex{2}
            @test issetequal(name.(Tuple(I)), (:i, :j))
            nc[I] = na[I] + nb[I]
        end
        @test unname(nc, (:i, :j)) ≈ unname(na, (:i, :j)) + unname(nb, (:i, :j))

        a = nameddims(randn(elt, 2, 3), (:i, :j))
        b = nameddims(randn(elt, 3, 2), (:j, :i))
        c = a + b
        @test unname(c, (:i, :j)) ≈ unname(a, (:i, :j)) + unname(b, (:i, :j))
        c = a .+ b
        @test unname(c, (:i, :j)) ≈ unname(a, (:i, :j)) + unname(b, (:i, :j))
        c = map(+, a, b)
        @test unname(c, (:i, :j)) ≈ unname(a, (:i, :j)) + unname(b, (:i, :j))
        c = nameddims(Array{elt}(undef, 2, 3), (:i, :j))
        c = map!(+, c, a, b)
        @test unname(c, (:i, :j)) ≈ unname(a, (:i, :j)) + unname(b, (:i, :j))
        c = a .+ 2 .* b
        @test unname(c, (:i, :j)) ≈ unname(a, (:i, :j)) + 2 * unname(b, (:i, :j))
        c = nameddims(Array{elt}(undef, 2, 3), (:i, :j))
        c .= a .+ 2 .* b
        @test unname(c, (:i, :j)) ≈ unname(a, (:i, :j)) + 2 * unname(b, (:i, :j))

        # Regression test for proper permutations.
        a = nameddims(randn(elt, 2, 3, 4), (:i, :j, :k))
        I = (:i => 2, :j => 3, :k => 4)
        for I′ in Combinatorics.permutations(I)
            @test a[I′...] == a[2, 3, 4]
            a′ = copy(a)
            a′[I′...] = zero(Bool)
            @test iszero(a′[2, 3, 4])
        end
        I = (:i => 2, :j => 2:3, :k => 4)
        for I′ in Combinatorics.permutations(I)
            @test a[I′...] == a[2, 2:3, 4]
            ## TODO: This is broken, investigate.
            ## a′[I′...] = zeros(Bool, 2)
            ## @test iszero(a′[2, 2:3, 4])
        end
    end
    @testset "conj/fill! (eltype=$elt)" for elt in TestBasicsUtils.elts
        i, j = namedoneto.((2, 3), ("i", "j"))
        a = randn(elt, i, j)

        # `conj` forwards to the underlying so that, on graded backends, sector
        # arrows on the axes flip. For plain ranges the axis-level `conj` is a
        # no-op and the test reduces to element-wise conjugation.
        ca = conj(a)
        @test unnamed(ca) == conj(unnamed(a))
        @test dimnames(ca) == dimnames(a)

        # `fill!` forwards to the underlying storage.
        b = randn(elt, i, j)
        @test fill!(b, zero(elt)) === b
        @test all(iszero, unnamed(b))
    end
    @testset "promote_leaf_eltypes (eltype=$elt)" for elt in TestBasicsUtils.elts
        # `LinearAlgebra.promote_leaf_eltypes` is called from `isapprox`; the
        # generic fallback iterates elements, which is expensive (or unsupported)
        # on block-structured backends. The named-array override delegates to
        # the underlying storage.
        i, j = namedoneto.((2, 3), ("i", "j"))
        a = randn(elt, i, j)
        @test LinearAlgebra.promote_leaf_eltypes(a) ===
            LinearAlgebra.promote_leaf_eltypes(unnamed(a))
    end
    @testset "sum/mapreduce (eltype=$elt)" for elt in TestBasicsUtils.elts
        # Reductions delegate to the underlying storage. `sum` is routed directly
        # rather than left to the generic `mapreduce` fallback because some backends
        # (such as graded arrays) define `Base.sum` without the general `mapreduce`,
        # so summing the unnamed data is the path that works for them.
        i, j = namedoneto.((2, 3), ("i", "j"))
        a = randn(elt, i, j)
        @test sum(a) == sum(unnamed(a))
        @test mapreduce(identity, +, a) == mapreduce(identity, +, unnamed(a))
    end
    @testset "begin/end (eltype=$elt)" for elt in TestBasicsUtils.elts
        i, j = namedoneto.((2, 3), ("i", "j"))
        a = randn(elt, i, j)
        @test a[begin, begin] == a[1, 1]
        @test a[2, begin] == a[2, 1]
        @test a[begin, 2] == a[1, 2]
        @test a[begin, end] == a[1, 3]
        @test a[end, begin] == a[2, 1]
        @test a[end, end] == a[2, 3]

        @test a[j => begin, i => begin] == a[1, 1]
        @test a[j => 2, i => begin] == a[1, 2]
        @test a[j => begin, i => 2] == a[2, 1]
        @test a[j => begin, i => end] == a[2, 1]
        @test a[j => end, i => begin] == a[1, 3]
        @test a[j => end, i => end] == a[2, 3]

        @test a[j[begin], i[begin]] == a[1, 1]
        @test a[j[2], i[begin]] == a[1, 2]
        @test a[j[begin], i[2]] == a[2, 1]
        @test a[j[begin], i[end]] == a[2, 1]
        @test a[j[end], i[begin]] == a[1, 3]
        @test a[j[end], i[end]] == a[2, 3]
    end
    @testset "Shorthand constructors (eltype=$elt)" for elt in TestBasicsUtils.elts
        i, j = named.((2, 2), ("i", "j"))
        value = rand(elt)
        for na in (zeros(elt, i, j), zeros(elt, (i, j)))
            @test eltype(na) ≡ elt
            @test na isa NamedTensor
            @test inds(na) == Base.oneto.([i, j])
            @test iszero(na)
        end
        for na in (fill(value, i, j), fill(value, (i, j)))
            @test eltype(na) ≡ elt
            @test na isa NamedTensor
            @test inds(na) == Base.oneto.([i, j])
            @test all(isequal(value), na)
        end
        for na in (rand(elt, i, j), rand(elt, (i, j)))
            @test eltype(na) ≡ elt
            @test na isa NamedTensor
            @test inds(na) == Base.oneto.([i, j])
            @test !iszero(na)
            @test all(x -> real(x) > 0, na)
        end
        for na in (randn(elt, i, j), randn(elt, (i, j)))
            @test eltype(na) ≡ elt
            @test na isa NamedTensor
            @test inds(na) == Base.oneto.([i, j])
            @test !iszero(na)
        end
    end
    @testset "Shorthand constructors (eltype=unspecified)" begin
        i, j = named.((2, 2), ("i", "j"))
        default_elt = Float64
        for na in (zeros(i, j), zeros((i, j)))
            @test eltype(na) ≡ default_elt
            @test na isa NamedTensor
            @test inds(na) == Base.oneto.([i, j])
            @test iszero(na)
        end
        for na in (rand(i, j), rand((i, j)))
            @test eltype(na) ≡ default_elt
            @test na isa NamedTensor
            @test inds(na) == Base.oneto.([i, j])
            @test !iszero(na)
            @test all(x -> real(x) > 0, na)
        end
        for na in (randn(i, j), randn((i, j)))
            @test eltype(na) ≡ default_elt
            @test na isa NamedTensor
            @test inds(na) == Base.oneto.([i, j])
            @test !iszero(na)
        end
    end
    @testset "show" begin
        a = NamedTensor([1 2; 3 4], ("i", "j"))
        @test sprint(show, "text/plain", a) ==
            "named(Base.OneTo(2), \"i\")×named(Base.OneTo(2), \"j\") " *
            "$NamedTensor{String}:\n" *
            "2×2 Matrix{Int64}:\n 1  2\n 3  4"

        a = NamedTensor([1 2; 3 4], ("i", "j"))
        @test sprint(show, a) ==
            "[1 2; 3 4][named(Base.OneTo(2), \"i\"), named(Base.OneTo(2), \"j\")]"
    end

    @testset "@names" begin
        x = @names x
        y, z = @names y z
        a, b, c = @names a[1:2] b[1:2, 1:2] c[2:3, [1, 2]]
        @test x == Name(:x)
        @test y == Name(:y)
        @test z == Name(:z)
        @test size(a) == (2,)
        @test a == [Name(:a_1), Name(:a_2)]
        @test size(b) == (2, 2)
        @test b == [
            Name(:b_1_1) Name(:b_1_2)
            Name(:b_2_1) Name(:b_2_2)
        ]
        @test size(c) == (2, 2)
        @test c == [
            Name(:c_2_1) Name(:c_2_2)
            Name(:c_3_1) Name(:c_3_2)
        ]
    end
end
