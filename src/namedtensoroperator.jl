using Base.Broadcast: Broadcast as BC, Broadcasted
using LinearAlgebra: LinearAlgebra as LA
using OrderedCollections: OrderedDict
using Random: Random

# Named dimension operator minimal interface.

# Choi state representation of the named operator.
# https://en.wikipedia.org/wiki/Choi%E2%80%93Jamio%C5%82kowski_isomorphism
"""
    state(a)

The underlying tensor of a named operator, with its codomain/domain structure
forgotten. An operator carries a tensor together with a pairing of its codomain and
domain dimension names (its Choi, or state, representation). `state` returns that tensor
on its own. For a plain tensor that is not an operator, `state` returns it unchanged.

# Examples

```jldoctest
julia> a = nameddims(zeros(2), (:i,));

julia> state(a) == a
true
```

See also [`operator`](@ref), [`codomainnames`](@ref), [`domainnames`](@ref).
"""
state(a) = throw(MethodError(state, (a,)))
# Operator representation of the named state given pairs of named codomain and domain indices.
operator(a, codomain, domain) = throw(MethodError(operator, (a, codomain, domain)))

# Get the codomain dimension names of the operator.
"""
    codomainnames(a)

The codomain (output) dimension names of an operator `a`. An operator pairs each of its
codomain names with a domain name. Applying the operator contracts over the domain and
leaves the codomain.

# Examples

```jldoctest
julia> op = operator(zeros(2, 2), ("i",), ("j",));

julia> collect(codomainnames(op))
1-element Vector{String}:
 "i"
```

See also [`domainnames`](@ref), [`operator`](@ref), [`apply`](@ref).
"""
codomainnames(a) = throw(MethodError(codomainnames, (a,)))

# Get the domain dimension names of the operator.
"""
    domainnames(a)

The domain (input) dimension names of an operator `a`. These are the names contracted
over when the operator is applied to a tensor.

# Examples

```jldoctest
julia> op = operator(zeros(2, 2), ("i",), ("j",));

julia> collect(domainnames(op))
1-element Vector{String}:
 "j"
```

See also [`codomainnames`](@ref), [`operator`](@ref), [`apply`](@ref).
"""
domainnames(a) = throw(MethodError(domainnames, (a,)))

# Given a domain dimension name, return the corresponding codomain dimension name.
# If it doesn't exist, return the index itself.
get_codomain_name(a, i) = throw(MethodError(get_codomain_name, (a, i)))
# Given a codomain dimension name, return the corresponding domain dimension name.
# If it doesn't exist, return the index itself.
get_domain_name(a, i) = throw(MethodError(get_domain_name, (a, i)))

"""
    apply(x::AbstractNamedTensor, y::AbstractNamedTensor)

Apply the operator `x` to `y`. This contracts the state tensors of `x` and `y` over
their shared names, then renames each surviving codomain name of `x` back to its paired
domain name, so the result carries the same names `y` would map to. Applying the
identity operator leaves `y` unchanged.

# Examples

```jldoctest
julia> op = operator(reshape(Float64[1, 0, 0, 1], 2, 2), ("i",), ("j",));

julia> v = nameddims([3.0, 4.0], ("j",));

julia> apply(op, v) == v
true
```

See also [`operator`](@ref), [`state`](@ref), [`codomainnames`](@ref),
[`domainnames`](@ref).
"""
function apply(x::AbstractNamedTensor, y::AbstractNamedTensor)
    xy = state(x) * state(y)
    return mapdimnames(xy) do i
        return get_domain_name(x, i)
    end
end

function apply_dag(x::AbstractNamedTensor, y::AbstractNamedTensor)
    xy = state(x) * state(y)
    return mapdimnames(xy) do i
        return get_codomain_name(y, i)
    end
end

# TODO: Define versions that accept codomain and domain names,
# i.e. `transpose(a, codomain, domain)` and `adjoint(a, codomain, domain)` (?).
function Base.transpose(a::AbstractNamedTensor)
    c = codomainnames(a)
    d = domainnames(a)
    a_map = merge(Dict(c .=> d), Dict(d .=> c))
    a′ = mapdimnames(state(a)) do i
        return get(a_map, i, i)
    end
    return operator(a′, c, d)
end
function Base.adjoint(a::AbstractNamedTensor)
    return transpose(conj(a))
end

function product(x::AbstractNamedTensor, y::AbstractNamedTensor)
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

struct NamedTensorOperator{DimName, P <: AbstractNamedTensor{DimName}, D, C} <:
    AbstractNamedTensor{DimName}
    parent::P
    dimnames_bijection::Bijection{D, C}
end

state(a::AbstractNamedTensor) = a
state(a::NamedTensorOperator) = a.parent
Base.parent(a::NamedTensorOperator) = state(a)
unnamed(a::NamedTensorOperator) = unnamed(state(a))
dimnames(a::NamedTensorOperator) = dimnames(state(a))

function NamedTensorOperator(a::AbstractNamedTensor, codomainnames, domainnames)
    return NamedTensorOperator(a, Bijection(domainnames, codomainnames))
end

parenttype(type::Type{<:NamedTensorOperator}) = fieldtype(type, :parent)
statetype(type::Type{<:NamedTensorOperator}) = parenttype(type)

function nameddimsof(a::NamedTensorOperator, b::AbstractArray)
    return NamedTensorOperator(nameddimsof(state(a), b), a.dimnames_bijection)
end

codomainnames(a::NamedTensorOperator) = codomain(a.dimnames_bijection)
domainnames(a::NamedTensorOperator) = domain(a.dimnames_bijection)

function get_codomain_name(a::NamedTensorOperator, i)
    return get(a.dimnames_bijection, i, i)
end
function get_domain_name(a::NamedTensorOperator, i)
    return get(inverse(a.dimnames_bijection), i, i)
end

"""
    operator(a, codomain, domain)

Build a named operator from a tensor (or plain array) `a` by partitioning its dimension
names into a `codomain` (output) set and a `domain` (input) set. The operator pairs each
codomain name with a domain name, so it can be applied to a tensor with
[`apply`](@ref), contracting over the domain. `codomain` and `domain` may be given as
dimension names or as named ranges such as `Index`es. Recover the underlying tensor
with [`state`](@ref) and the name sets with [`codomainnames`](@ref) and
[`domainnames`](@ref).

# Examples

```jldoctest
julia> op = operator(zeros(2, 2), ("i",), ("j",));

julia> collect(codomainnames(op))
1-element Vector{String}:
 "i"

julia> collect(domainnames(op))
1-element Vector{String}:
 "j"
```

See also [`state`](@ref), [`codomainnames`](@ref), [`domainnames`](@ref),
[`apply`](@ref), [`similar_operator`](@ref).
"""
function operator end

# `codomain` and `domain` may be given as dimension names or as named ranges
# (such as `Index`es); `name` maps the latter to their names and leaves names as-is.
# TODO: Unify these two functions.
function operator(a::AbstractArray, codomain, domain)
    codomain, domain = name.(codomain), name.(domain)
    na = nameddims(a, (codomain..., domain...))
    return operator(na, codomain, domain)
end
function operator(a::AbstractNamedTensor, codomain, domain)
    return NamedTensorOperator(a, name.(codomain), name.(domain))
end

# Operator-preserving contraction. Contracting two named arrays sums over their
# shared names, so the result keeps each operand's surviving codomain/domain
# structure. A non-operator tensor contributes no pairs (all its names are
# dangling from the operator point of view). The result is always an
# `NamedTensorOperator`, even when its codomain and domain both come out empty, so
# the product type does not depend on the runtime names being contracted.
operator_pairs(a::NamedTensorOperator) = a.dimnames_bijection.domain_to_codomain
operator_pairs(a::AbstractNamedTensor) = ()

# Compose the codomain/domain of `a * b`. The `domain => codomain` pairs of both
# operands form a graph of maximum degree two (each name is paired at most once
# per operand), so it is a disjoint union of simple paths. Each surviving domain
# name reaches its surviving codomain partner by following the pairing through
# any contracted (shared) names. A name whose chain dead-ends on a contracted
# index is left dangling, so the result is well defined for any contraction.
function product_codomain_domain(a::AbstractNamedTensor, b::AbstractNamedTensor)
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

function operator_product(a::AbstractNamedTensor, b::AbstractNamedTensor)
    ab = state(a) * state(b)
    codomain, domain = product_codomain_domain(a, b)
    return operator(ab, codomain, domain)
end

Base.:*(a::NamedTensorOperator, b::NamedTensorOperator) = operator_product(a, b)
Base.:*(a::NamedTensorOperator, b::AbstractNamedTensor) = operator_product(a, b)
Base.:*(a::AbstractNamedTensor, b::NamedTensorOperator) = operator_product(a, b)

# Operator-preserving broadcasting (the style struct and style-combination rules
# live in `broadcast.jl`). An `NamedTensorOperator` broadcasts as itself, so `op .+ op`,
# `2 .* op`, etc. carry `NamedTensorOperatorStyle`; `+` / `-` / scalar `*` inherit
# preservation since they lower to broadcasting. `copy` / `similar` unwrap each
# operator operand to its `state` (the shared `NamedTensorStyle` machinery does this via
# `unnamed`), build the `NamedTensor` result, then rewrap as an operator using the
# codomain/domain split recovered from the operands.
function BC.BroadcastStyle(arraytype::Type{<:NamedTensorOperator})
    return NamedTensorOperatorStyle{ndims(arraytype)}()
end

# Recover the codomain/domain split shared by all operator operands of `bc`,
# erroring if any two operators disagree.
operator_operands(bc::Broadcasted) = operator_operands(bc.args...)
function operator_operands(arg::NamedTensorOperator, args...)
    return (arg, operator_operands(args...)...)
end
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

function Base.copy(bc::Broadcasted{<:NamedTensorOperatorStyle})
    cod, dom = broadcast_operator_codomain_domain(bc)
    result = copy(statebroadcasted(bc))
    return operator(result, cod, dom)
end

for f in MATRIX_FUNCTIONS
    @eval begin
        function Base.$f(a::NamedTensorOperator)
            c = codomainnames(a)
            d = domainnames(a)
            return operator($f(state(a), c, d), c, d)
        end
    end
end

# Operator entries for the gram factorizations defined in `tensoralgebra.jl`.
# Placed here because `NamedTensorOperator` is defined in this file, which comes
# after `tensoralgebra.jl` in the include order.
#
# Per-method docstrings are factored out into `const` strings and attached
# inside the `@eval` loop via `@doc`. This keeps the loop body uniform when
# methods need distinct user-facing docs (including jldoctest examples) that
# don't share enough structure to warrant `$($f)`-interpolation.

const _gram_eigh_full_operator_docstring = """
    TensorAlgebra.MatrixAlgebra.gram_eigh_full(a::NamedTensorOperator; kwargs...) -> x

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
    TensorAlgebra.MatrixAlgebra.gram_eigh_full_with_pinv(a::NamedTensorOperator; kwargs...) -> x, y

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
        @doc $doc_sym function MA.$f(a::NamedTensorOperator; kwargs...)
            return MA.$f(state(a), codomainnames(a), domainnames(a); kwargs...)
        end
    end
end

# Operator forms of the Hermitian-square-root family: the codomain/domain partition is
# taken from the operator, so callers can compose with `transpose` to choose the
# bipartition — e.g. project Hermitian in one bipartition and take the square root in the
# transposed one, which induces the fermionic braid sign on the odd-parity sector.
# `project_hermitian` returns an operator (its Hermitian part is again a bond operator); the
# roots return bare named arrays like `gram_eigh_full`, being terminal factorization outputs.
function MAK.project_hermitian(a::NamedTensorOperator; kwargs...)
    h = MAK.project_hermitian(state(a), codomainnames(a), domainnames(a); kwargs...)
    return operator(h, codomainnames(a), domainnames(a))
end
for f in (:sqrth_safe, :invsqrth_safe, :sqrth_invsqrth_safe)
    @eval function MA.$f(a::NamedTensorOperator; kwargs...)
        return MA.$f(state(a), codomainnames(a), domainnames(a); kwargs...)
    end
end

"""
    Base.one(op::NamedTensorOperator) -> Id

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
function Base.one(op::NamedTensorOperator)
    co, dom = codomainnames(op), domainnames(op)
    return operator(one(state(op), co, dom), co, dom)
end

"""
    LinearAlgebra.tr(op::NamedTensorOperator) -> scalar

Trace of a named operator: contracts each codomain index with its paired domain index and
sums the diagonal, over the operator's intrinsic codomain/domain split.
"""
function LA.tr(op::NamedTensorOperator)
    a = state(op)
    byname = Dict(name(i) => i for i in inds(a))
    codomain = map(n -> byname[n], Tuple(codomainnames(op)))
    domain = map(n -> byname[n], Tuple(domainnames(op)))
    return LA.tr(a, codomain, domain)
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

# Examples

```jldoctest
julia> op = similar_operator(zeros(2, 2), (Base.OneTo(2),), (:i,), (:j,));

julia> collect(domainnames(op))
1-element Vector{Symbol}:
 :j
```

See also [`operator`](@ref), [`uniquename`](@ref).
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
# itself peels to the concrete storage via the generic AbstractNamedTensor
# method.

function Random.randn!(rng::Random.AbstractRNG, op::NamedTensorOperator)
    Random.randn!(rng, state(op))
    return op
end

function Random.rand!(rng::Random.AbstractRNG, op::NamedTensorOperator)
    Random.rand!(rng, state(op))
    return op
end
