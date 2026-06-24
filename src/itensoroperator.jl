using Base.Broadcast: Broadcast as BC, Broadcasted
using OrderedCollections: OrderedDict
using Random: Random

# Named dimension operator minimal interface.

# Choi state representation of the named operator.
# https://en.wikipedia.org/wiki/Choi%E2%80%93Jamio%C5%82kowski_isomorphism
state(a) = throw(MethodError(state, (a,)))
# Operator representation of the named state given pairs of named codomain and domain indices.
operator(a, codomain, domain) = throw(MethodError(operator, (a, codomain, domain)))

# Get the codomain dimension names of the operator.
codomainnames(a) = throw(MethodError(codomainnames, (a,)))
# Get the domain dimension names of the operator.
domainnames(a) = throw(MethodError(domainnames, (a,)))

# Given a domain dimension name, return the corresponding codomain dimension name.
# If it doesn't exist, return the index itself.
get_codomain_name(a, i) = throw(MethodError(get_codomain_name, (a, i)))
# Given a codomain dimension name, return the corresponding domain dimension name.
# If it doesn't exist, return the index itself.
get_domain_name(a, i) = throw(MethodError(get_domain_name, (a, i)))

function apply(x::AbstractITensor, y::AbstractITensor)
    xy = state(x) * state(y)
    return mapdimnames(xy) do i
        return get_domain_name(x, i)
    end
end

function apply_dag(x::AbstractITensor, y::AbstractITensor)
    xy = state(x) * state(y)
    return mapdimnames(xy) do i
        return get_codomain_name(y, i)
    end
end

# TODO: Define versions that accept codomain and domain names,
# i.e. `transpose(a, codomain, domain)` and `adjoint(a, codomain, domain)` (?).
function Base.transpose(a::AbstractITensor)
    c = codomainnames(a)
    d = domainnames(a)
    a_map = merge(Dict(c .=> d), Dict(d .=> c))
    a′ = mapdimnames(state(a)) do i
        return get(a_map, i, i)
    end
    return operator(a′, c, d)
end
function Base.adjoint(a::AbstractITensor)
    return transpose(conj(a))
end

function product(x::AbstractITensor, y::AbstractITensor)
    c = codomainnames(x)
    d = domainnames(x)
    c′ = uniquename.(c)
    x′_map = merge(Dict(c .=> c′), Dict(d .=> c))
    x′ = mapdimnames(parent(x)) do i
        return get(x′_map, i, i)
    end
    x′y = x′ * parent(y)
    x′y_map = Dict(c′ .=> c)
    xy = mapdimnames(x′y) do i
        return get(x′y_map, i, i)
    end
    return operator(xy, c, d)
end

struct Bijection{Codomain, Domain} <: AbstractDict{Domain, Codomain}
    domain_to_codomain::OrderedDict{Domain, Codomain}
    codomain_to_domain::OrderedDict{Codomain, Domain}
end
function Bijection(domain, codomain)
    pairs = domain .=> codomain
    domain_to_codomain = OrderedDict(pairs)
    codomain_to_domain = OrderedDict(reverse(kv) for kv in pairs)
    return Bijection(domain_to_codomain, codomain_to_domain)
end
function Base.get(b::Bijection, k, default)
    return get(b.domain_to_codomain, k, default)
end
function inverse(b::Bijection)
    return Bijection(b.codomain_to_domain, b.domain_to_codomain)
end
# Both accessors return the `keys(::OrderedDict)` of the dict that has the
# requested side as its key type, so the result is a `Base.KeySet` that
# compares correctly with `==`. The two dicts are constructed from the same
# pairs in the constructor, so `codomain(b)[i]` and `domain(b)[i]` remain in
# lock-step positional order.
function codomain(b::Bijection)
    return keys(b.codomain_to_domain)
end
function domain(b::Bijection)
    return keys(b.domain_to_codomain)
end
Base.iterate(b::Bijection) = iterate(b.domain_to_codomain)
Base.iterate(b::Bijection, state) = iterate(b.domain_to_codomain, state)
Base.length(b::Bijection) = length(b.domain_to_codomain)

struct ITensorOperator{DimName, P <: AbstractITensor{DimName}, D, C} <:
    AbstractITensor{DimName}
    parent::P
    dimnames_bijection::Bijection{D, C}
end

state(a::AbstractITensor) = a
state(a::ITensorOperator) = a.parent
Base.parent(a::ITensorOperator) = state(a)
unnamed(a::ITensorOperator) = unnamed(state(a))
dimnames(a::ITensorOperator) = dimnames(state(a))

function ITensorOperator(a::AbstractITensor, codomainnames, domainnames)
    return ITensorOperator(a, Bijection(domainnames, codomainnames))
end

parenttype(type::Type{<:ITensorOperator}) = fieldtype(type, :parent)
statetype(type::Type{<:ITensorOperator}) = parenttype(type)

function nameddimsof(a::ITensorOperator, b::AbstractArray)
    return ITensorOperator(nameddimsof(state(a), b), a.dimnames_bijection)
end

codomainnames(a::ITensorOperator) = codomain(a.dimnames_bijection)
domainnames(a::ITensorOperator) = domain(a.dimnames_bijection)

function get_codomain_name(a::ITensorOperator, i)
    return get(a.dimnames_bijection, i, i)
end
function get_domain_name(a::ITensorOperator, i)
    return get(inverse(a.dimnames_bijection), i, i)
end

# TODO: Unify these two functions.
function operator(a::AbstractArray, codomain, domain)
    na = nameddims(a, (codomain..., domain...))
    return operator(na, codomain, domain)
end
function operator(a::AbstractITensor, codomain, domain)
    return ITensorOperator(a, codomain, domain)
end

# Operator-preserving contraction. Contracting two named arrays sums over their
# shared names, so the result keeps each operand's surviving codomain/domain
# structure. A non-operator tensor contributes no pairs (all its names are
# dangling from the operator point of view). The result is always an
# `ITensorOperator`, even when its codomain and domain both come out empty, so
# the product type does not depend on the runtime names being contracted.
operator_pairs(a::ITensorOperator) = a.dimnames_bijection.domain_to_codomain
operator_pairs(a::AbstractITensor) = ()

# Compose the codomain/domain of `a * b`. The `domain => codomain` pairs of both
# operands form a graph of maximum degree two (each name is paired at most once
# per operand), so it is a disjoint union of simple paths. Each surviving domain
# name reaches its surviving codomain partner by following the pairing through
# any contracted (shared) names. A name whose chain dead-ends on a contracted
# index is left dangling, so the result is well defined for any contraction.
function product_codomain_domain(a::AbstractITensor, b::AbstractITensor)
    shared = intersect(dimnames(a), dimnames(b))
    pairs = collect(Iterators.flatten((operator_pairs(a), operator_pairs(b))))
    forward = Dict(pairs)
    domain = eltype(keys(forward))[]
    codomain = eltype(values(forward))[]
    for (d, c) in pairs
        d in shared && continue
        while c in shared && haskey(forward, c)
            c = forward[c]
        end
        c in shared && continue
        push!(domain, d)
        push!(codomain, c)
    end
    return codomain, domain
end

function operator_product(a::AbstractITensor, b::AbstractITensor)
    ab = state(a) * state(b)
    codomain, domain = product_codomain_domain(a, b)
    return operator(ab, codomain, domain)
end

Base.:*(a::ITensorOperator, b::ITensorOperator) = operator_product(a, b)
Base.:*(a::ITensorOperator, b::AbstractITensor) = operator_product(a, b)
Base.:*(a::AbstractITensor, b::ITensorOperator) = operator_product(a, b)

# Operator-preserving broadcasting (the style struct and style-combination rules
# live in `broadcast.jl`). An `ITensorOperator` broadcasts as itself, so `op .+ op`,
# `2 .* op`, etc. carry `ITensorOperatorStyle`; `+` / `-` / scalar `*` inherit
# preservation since they lower to broadcasting. `copy` / `similar` unwrap each
# operator operand to its `state` (the shared `ITensorStyle` machinery does this via
# `unnamed`), build the `ITensor` result, then rewrap as an operator using the
# codomain/domain split recovered from the operands.
function BC.BroadcastStyle(arraytype::Type{<:ITensorOperator})
    return ITensorOperatorStyle{ndims(arraytype)}()
end

# Recover the codomain/domain split shared by all operator operands of `bc`,
# erroring if any two operators disagree.
operator_operands(bc::Broadcasted) = operator_operands(bc.args...)
operator_operands(arg::ITensorOperator, args...) = (arg, operator_operands(args...)...)
function operator_operands(arg::Broadcasted, args...)
    return (operator_operands(arg.args...)..., operator_operands(args...)...)
end
operator_operands(arg, args...) = operator_operands(args...)
operator_operands() = ()

function broadcast_operator_codomain_domain(bc::Broadcasted)
    ops = operator_operands(bc)
    op1 = first(ops)
    cod1 = codomainnames(op1)
    dom1 = domainnames(op1)
    for op in Base.tail(ops)
        (issetequal(codomainnames(op), cod1) && issetequal(domainnames(op), dom1)) ||
            throw(
            ArgumentError(
                "Operator operands disagree on their codomain/domain split: " *
                    "$((cod1, dom1)) vs $((codomainnames(op), domainnames(op))). " *
                    "Broadcasting operators requires a matching split."
            )
        )
    end
    return cod1, dom1
end

function Base.copy(bc::Broadcasted{<:ITensorOperatorStyle})
    cod, dom = broadcast_operator_codomain_domain(bc)
    result = copy(statebroadcasted(bc))
    return operator(result, cod, dom)
end

function Base.similar(bc::Broadcasted{<:ITensorOperatorStyle}, elt::Type, ax)
    cod, dom = broadcast_operator_codomain_domain(bc)
    result = similar(statebroadcasted(bc), elt, ax)
    return operator(result, cod, dom)
end

for f in MATRIX_FUNCTIONS
    @eval begin
        function Base.$f(a::ITensorOperator)
            c = codomainnames(a)
            d = domainnames(a)
            return operator($f(state(a), c, d), c, d)
        end
    end
end

# Operator entries for the gram factorizations defined in `tensoralgebra.jl`.
# Placed here because `ITensorOperator` is defined in this file, which comes
# after `tensoralgebra.jl` in the include order.
#
# Per-method docstrings are factored out into `const` strings and attached
# inside the `@eval` loop via `@doc`. This keeps the loop body uniform when
# methods need distinct user-facing docs (including jldoctest examples) that
# don't share enough structure to warrant `$($f)`-interpolation.

const _gram_eigh_full_operator_docstring = """
    TensorAlgebra.MatrixAlgebra.gram_eigh_full(a::ITensorOperator; kwargs...) -> x

Gram factorization of a Hermitian positive semi-definite named operator
`a`, returning `x` such that `x * x_cod ≈ state(a)`, where `x_cod` is
`conj(x)` with its domain dimension names replaced by the corresponding
codomain names of `a`. `x` carries `a`'s domain dimension names and a
fresh trailing rank name. The codomain and domain partition is taken from
`codomainnames(a)` and `domainnames(a)`.

`kwargs` are forwarded to `TensorAlgebra.MatrixAlgebra.gram_eigh_full` on the
underlying named array (e.g. `atol`, `rtol`).

# Examples

```jldoctest
julia> using ITensorBase: namedoneto, operator, replacedimnames, state

julia> using TensorAlgebra.MatrixAlgebra: gram_eigh_full

julia> i, j, k, l, aux = namedoneto.((2, 2, 2, 2, 8), ("i", "j", "k", "l", "aux"));

julia> b = randn(aux, i, k);

julia> a = operator(conj(b) * replacedimnames(b, "i" => "j", "k" => "l"), ("i", "k"), ("j", "l"));

julia> x = gram_eigh_full(a);

julia> replacedimnames(x, "j" => "i", "l" => "k") * conj(x) ≈ state(a)
true
```
"""

const _gram_eigh_full_with_pinv_operator_docstring = """
    TensorAlgebra.MatrixAlgebra.gram_eigh_full_with_pinv(a::ITensorOperator; kwargs...) -> x, y

Like `TensorAlgebra.MatrixAlgebra.gram_eigh_full`, but additionally returns a
named array `y` that is a left inverse of `x`: `y * x ≈ I` on the
rank subspace (equal to the identity when `a` is full rank). The
codomain and domain partition is taken from `codomainnames(a)` and
`domainnames(a)`.

# Examples

```jldoctest
julia> using LinearAlgebra: I

julia> using ITensorBase: unname, dimnames, namedoneto, operator, replacedimnames

julia> using TensorAlgebra.MatrixAlgebra: gram_eigh_full_with_pinv

julia> i, j, k, l, aux = namedoneto.((2, 2, 2, 2, 8), ("i", "j", "k", "l", "aux"));

julia> b = randn(aux, i, k);

julia> a = operator(conj(b) * replacedimnames(b, "i" => "j", "k" => "l"), ("i", "k"), ("j", "l"));

julia> x, y = gram_eigh_full_with_pinv(a);

julia> rname = only(setdiff(dimnames(x), ("j", "l")));

julia> reshape(unname(y, (rname, "j", "l")), :, 4) *
       reshape(unname(x, ("j", "l", rname)), 4, :) ≈ I
true
```
"""

for f in (:gram_eigh_full, :gram_eigh_full_with_pinv)
    doc_sym = Symbol("_", f, "_operator_docstring")
    @eval begin
        @doc $doc_sym function MA.$f(a::ITensorOperator; kwargs...)
            return MA.$f(state(a), codomainnames(a), domainnames(a); kwargs...)
        end
    end
end

"""
    Base.one(op::ITensorOperator) -> Id

Return the identity operator with the same codomain/domain names and shape as
`op`. `op` is treated as a shape prototype and is not mutated.

The identity acts as the multiplicative identity for `ITensorBase.apply`: it
contracts on the domain names and renames the resulting codomain names back to
the domain names, leaving the input unchanged.

# Examples

```jldoctest
julia> using ITensorBase: apply, namedoneto, operator

julia> i, j, k, l = namedoneto.((2, 3, 2, 3), ("i", "j", "k", "l"));

julia> op = operator(randn(i, j, k, l), ("i", "j"), ("k", "l"));

julia> Id = one(op);

julia> v = randn(k, l);

julia> apply(Id, v) ≈ v
true
```
"""
function Base.one(op::ITensorOperator)
    co, dom = codomainnames(op), domainnames(op)
    return operator(one(state(op), co, dom), co, dom)
end

# === similar_operator ===
#
# Allocate an operator with the user-supplied axes as the domain (input). The
# codomain (output) shares the domain direction and either takes
# explicitly-supplied names or fresh `uniquename` outputs. The 5-arg form is
# canonical, the others fill in defaults. The bra/ket flip on the storage side
# is handled inside `TA.similar_map`.

"""
    similar_operator(prototype, [T,] unnamed_domain_axes, [codomain_names,] domain_names) -> op
    similar_operator(prototype, [T,] named_domain_axes) -> op

Allocate an operator-shaped named array with undefined data, with the
user-supplied side as the domain (input) and a matching codomain (output).
Element type defaults to `eltype(prototype)`. Codomain names default to fresh
`uniquename`-generated names. The first form takes unnamed (raw) axes and
explicit names, the second takes already-named axes and reuses their names as
the domain. Storage layout (including the bra/ket flip on the domain side for
graded axes) is delegated to `TensorAlgebra.similar_map`.
"""
function similar_operator(
        prototype, ::Type{T}, unnamed_domain_axes, codomain_names, domain_names
    ) where {T}
    codomain_axes = named.(unnamed_domain_axes, codomain_names)
    domain_axes = named.(unnamed_domain_axes, domain_names)
    raw = TA.similar_map(prototype, T, codomain_axes, domain_axes)
    return operator(raw, codomain_names, domain_names)
end
function similar_operator(
        prototype, ::Type{T}, unnamed_domain_axes, domain_names
    ) where {T}
    codomain_names = uniquename.(domain_names)
    return similar_operator(
        prototype, T, unnamed_domain_axes, codomain_names, domain_names
    )
end
function similar_operator(prototype, ::Type{T}, named_domain_axes) where {T}
    return similar_operator(
        prototype, T, unnamed.(named_domain_axes), name.(named_domain_axes)
    )
end
function similar_operator(prototype, unnamed_domain_axes, codomain_names, domain_names)
    return similar_operator(
        prototype, eltype(prototype), unnamed_domain_axes, codomain_names, domain_names
    )
end
function similar_operator(prototype, unnamed_domain_axes, domain_names)
    return similar_operator(prototype, eltype(prototype), unnamed_domain_axes, domain_names)
end
function similar_operator(prototype, named_domain_axes)
    return similar_operator(prototype, eltype(prototype), named_domain_axes)
end

# Forward `Random.randn!` / `Random.rand!` to the operator's state, which
# itself peels to the concrete storage via the generic AbstractITensor
# method.

function Random.randn!(rng::Random.AbstractRNG, op::ITensorOperator)
    Random.randn!(rng, state(op))
    return op
end

function Random.rand!(rng::Random.AbstractRNG, op::ITensorOperator)
    Random.rand!(rng, state(op))
    return op
end
