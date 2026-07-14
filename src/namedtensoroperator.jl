using Base.Broadcast: Broadcast as BC, Broadcasted
using LinearAlgebra: LinearAlgebra as LA
using Random: Random

# Named dimension operator minimal interface.

# Choi state representation of the named operator.
# https://en.wikipedia.org/wiki/Choi%E2%80%93Jamio%C5%82kowski_isomorphism
"""
    state(a)

The underlying tensor of a named operator, with its output/input structure
forgotten. An operator carries a tensor together with a pairing of its output and
input dimension names (its Choi, or state, representation). `state` returns that tensor
on its own. For a plain tensor that is not an operator, `state` returns it unchanged.

# Examples

```jldoctest
julia> a = nameddims(zeros(2), (:i,));

julia> state(a) == a
true
```

See also [`operator`](@ref), [`outputnames`](@ref), [`inputnames`](@ref).
"""
state(a) = throw(MethodError(state, (a,)))
# Operator representation of the named state given pairs of named output and input indices.
operator(a, output, input) = throw(MethodError(operator, (a, output, input)))

# Get the output dimension names of the operator.
"""
    outputnames(a)

The output dimension names of an operator `a`. An operator pairs each of its
output names with an input name. Applying the operator contracts over the input and
leaves the output. A plain tensor is a trivial operator with no pairing, so its output
names are empty.

# Examples

```jldoctest
julia> op = operator(zeros(2, 2), ("i",), ("j",));

julia> collect(outputnames(op))
1-element Vector{String}:
 "i"
```

See also [`inputnames`](@ref), [`operator`](@ref), [`apply`](@ref).
"""
outputnames(a::AbstractNamedTensor) = ()

# Get the input dimension names of the operator.
"""
    inputnames(a)

The input dimension names of an operator `a`. These are the names contracted
over when the operator is applied to a tensor. A plain tensor is a trivial operator with
no pairing, so its input names are empty.

# Examples

```jldoctest
julia> op = operator(zeros(2, 2), ("i",), ("j",));

julia> collect(inputnames(op))
1-element Vector{String}:
 "j"
```

See also [`outputnames`](@ref), [`operator`](@ref), [`apply`](@ref).
"""
inputnames(a::AbstractNamedTensor) = ()

# `outputname(a, i, default)` returns the output name paired with input name `i`, and
# `inputname(a, i, default)` returns the input name paired with output name `i`, each
# returning `default` when `i` is not one of the operator's paired names (mirroring
# `Base.get`). A tensor with no operator pairing returns `default` for every name.
outputname(a::AbstractNamedTensor, i, default) = default
inputname(a::AbstractNamedTensor, i, default) = default

"""
    apply(x::AbstractNamedTensor, y::AbstractNamedTensor)

Apply the operator `x` to `y`. This contracts the state tensors of `x` and `y` over
their shared names, then renames each surviving output name of `x` back to its paired
input name, so the result carries the same names `y` would map to. Applying the
identity operator leaves `y` unchanged.

# Examples

```jldoctest
julia> op = operator(reshape(Float64[1, 0, 0, 1], 2, 2), ("i",), ("j",));

julia> v = nameddims([3.0, 4.0], ("j",));

julia> apply(op, v) == v
true
```

See also [`operator`](@ref), [`state`](@ref), [`outputnames`](@ref),
[`inputnames`](@ref).
"""
function apply(x::AbstractNamedTensor, y::AbstractNamedTensor)
    xy = state(x) * state(y)
    return mapdimnames(xy) do i
        return inputname(x, i, i)
    end
end

function apply_dag(x::AbstractNamedTensor, y::AbstractNamedTensor)
    xy = state(x) * state(y)
    return mapdimnames(xy) do i
        return outputname(y, i, i)
    end
end

# TODO: Define versions that accept output and input names,
# i.e. `transpose(a, output, input)` and `adjoint(a, output, input)` (?).
function Base.transpose(a::AbstractNamedTensor)
    out = outputnames(a)
    inp = inputnames(a)
    a_map = merge(Dict(out .=> inp), Dict(inp .=> out))
    a′ = mapdimnames(state(a)) do i
        return get(a_map, i, i)
    end
    return operator(a′, out, inp)
end
function Base.adjoint(a::AbstractNamedTensor)
    return transpose(conj(a))
end

# A wire of `a` and a wire of `b` may either be the same wire (composed by `product`) or
# share no name (independent). Any other overlap is rejected: two different wires ending on
# the same input site or leaving the same output site is ambiguous, and connecting two
# operators end-to-end through a shared bond is a job for `*`, not `product`.
function check_product(a::AbstractNamedTensor, b::AbstractNamedTensor)
    for (out_a, in_a) in zip(outputnames(a), inputnames(a)),
            (out_b, in_b) in zip(outputnames(b), inputnames(b))

        overlap = !isdisjoint((out_a, in_a), (out_b, in_b))
        match = (out_a == out_b) && (in_a == in_b)
        if overlap && !match
            throw(
                ArgumentError(
                    "`product` operands have overlapping but mismatched wires " *
                        "($out_a←$in_a vs $out_b←$in_b); wires must match to compose or share " *
                        "no name. Use `*` to connect operators end-to-end."
                )
            )
        end
    end
    return nothing
end

# `product(a, b)` composes operators on matching sites. A name that is an input of both
# operands marks a shared site: its (matching) wire is welded — `a`'s input and `b`'s
# output are renamed to a fresh bond — so the site composes instead of contracting
# input↔input. Every other shared name — dangling Kraus/batch dimensions, a plain tensor's
# legs — contracts like `*`. See `check_product` for the wires that are rejected.
function product(a::AbstractNamedTensor, b::AbstractNamedTensor)
    check_product(a, b)
    a′, b′ = a, b
    for s in intersect(inputnames(a), inputnames(b))
        bond = uniquename(s)
        a′ = replacedimnames(a′, s => bond)                      # a's input site → bond
        b′ = replacedimnames(b′, outputname(b, s, s) => bond)   # b's matching output → bond
    end
    return operator_product(a′, b′)
end

struct NamedTensorOperator{DimName, P <: AbstractNamedTensor{DimName}} <:
    AbstractNamedTensor{DimName}
    parent::P
    # `outputnames[i]` is paired with `inputnames[i]` (positional pairing), so the two
    # vectors always have equal length. Names in the parent that appear in neither vector
    # are dangling (not part of the operator pairing).
    outputnames::Vector{DimName}
    inputnames::Vector{DimName}
end

function NamedTensorOperator(
        parent::AbstractNamedTensor{DimName}, outputnames, inputnames
    ) where {DimName}
    if length(outputnames) != length(inputnames)
        throw(
            ArgumentError(
                "Operator `outputnames` and `inputnames` must have equal length " *
                    "(positional pairing), got $(length(outputnames)) and " *
                    "$(length(inputnames))."
            )
        )
    end
    return NamedTensorOperator{DimName, typeof(parent)}(
        parent, collect(DimName, outputnames), collect(DimName, inputnames)
    )
end

state(a::AbstractNamedTensor) = a
state(a::NamedTensorOperator) = a.parent
Base.parent(a::NamedTensorOperator) = state(a)
unnamed(a::NamedTensorOperator) = unnamed(state(a))
dimnames(a::NamedTensorOperator) = dimnames(state(a))

parenttype(type::Type{<:NamedTensorOperator}) = fieldtype(type, :parent)
statetype(type::Type{<:NamedTensorOperator}) = parenttype(type)

function nameddimsof(a::NamedTensorOperator, b::AbstractArray)
    return NamedTensorOperator(nameddimsof(state(a), b), a.outputnames, a.inputnames)
end

outputnames(a::NamedTensorOperator) = a.outputnames
inputnames(a::NamedTensorOperator) = a.inputnames

# Relabeling an operator's dimension names updates both its state and its pairing (the
# generic `AbstractNamedTensor` methods reconstruct via `nameddims` and would drop the
# pairing). `mapdimnames(f, op)` routes through the function form via the generic
# `mapdimnames(f, ::AbstractNamedTensor) = replacedimnames(f, ...)`.
function replacedimnames(op::NamedTensorOperator, replacements::Pair...)
    isempty(replacements) && return op
    ps = map(p -> name(first(p)) => name(last(p)), replacements)
    return operator(
        replacedimnames(state(op), ps...),
        replace(outputnames(op), ps...),
        replace(inputnames(op), ps...)
    )
end
function replacedimnames(f, op::NamedTensorOperator)
    return operator(
        replacedimnames(f, state(op)), map(f, outputnames(op)), map(f, inputnames(op))
    )
end

# Linear scan over the paired name vectors: for the small number of paired legs an
# operator carries, a `findfirst` is faster than maintaining a hashed lookup, and it
# keeps the operator representation to just the two vectors.
function outputname(a::NamedTensorOperator, i, default)
    k = findfirst(==(i), a.inputnames)
    return isnothing(k) ? default : a.outputnames[k]
end
function inputname(a::NamedTensorOperator, i, default)
    k = findfirst(==(i), a.outputnames)
    return isnothing(k) ? default : a.inputnames[k]
end

"""
    operator(a, output, input)

Build a named operator from a tensor (or plain array) `a` by partitioning its dimension
names into an `output` set and an `input` set. The operator pairs each output name with
an input name, so it can be applied to a tensor with [`apply`](@ref), contracting over
the input. `output` and `input` may be given as dimension names or as named ranges such
as `Index`es. Recover the underlying tensor with [`state`](@ref) and the name sets with
[`outputnames`](@ref) and [`inputnames`](@ref).

# Examples

```jldoctest
julia> op = operator(zeros(2, 2), ("i",), ("j",));

julia> collect(outputnames(op))
1-element Vector{String}:
 "i"

julia> collect(inputnames(op))
1-element Vector{String}:
 "j"
```

See also [`state`](@ref), [`outputnames`](@ref), [`inputnames`](@ref),
[`apply`](@ref), [`similar_operator`](@ref).
"""
function operator end

# `output` and `input` may be given as dimension names or as named ranges
# (such as `Index`es); `name` maps the latter to their names and leaves names as-is.
# TODO: Unify these two functions.
function operator(a::AbstractArray, output, input)
    output, input = name.(output), name.(input)
    na = nameddims(a, (output..., input...))
    return operator(na, output, input)
end
function operator(a::AbstractNamedTensor, output, input)
    return NamedTensorOperator(a, name.(output), name.(input))
end

# Operator-preserving contraction. Contracting two named arrays sums over their
# shared names, so the result keeps each operand's surviving output/input
# structure. A non-operator tensor contributes no pairs (all its names are
# dangling from the operator point of view). The result is always an
# `NamedTensorOperator`, even when its output and input both come out empty, so
# the product type does not depend on the runtime names being contracted.
operator_pairs(a::NamedTensorOperator) = a.inputnames .=> a.outputnames
operator_pairs(a::AbstractNamedTensor) = ()

# Compose the output/input of `a * b`. The `input => output` pairs of both
# operands form a graph of maximum degree two (each name is paired at most once
# per operand), so it is a disjoint union of simple paths. Each surviving input
# name reaches its surviving output partner by following the pairing through
# any contracted (shared) names. A name whose chain dead-ends on a contracted
# index is left dangling, so the result is well defined for any contraction.
function product_output_input(a::AbstractNamedTensor, b::AbstractNamedTensor)
    shared = intersect(dimnames(a), dimnames(b))
    pairs = collect(Iterators.flatten((operator_pairs(a), operator_pairs(b))))
    forward = Dict(pairs)
    input = eltype(keys(forward))[]
    output = eltype(values(forward))[]
    for (inp, out) in pairs
        inp in shared && continue
        while out in shared && haskey(forward, out)
            out = forward[out]
        end
        out in shared && continue
        push!(input, inp)
        push!(output, out)
    end
    return output, input
end

function operator_product(a::AbstractNamedTensor, b::AbstractNamedTensor)
    ab = state(a) * state(b)
    output, input = product_output_input(a, b)
    return operator(ab, output, input)
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
# output/input split recovered from the operands.
function BC.BroadcastStyle(arraytype::Type{<:NamedTensorOperator})
    return NamedTensorOperatorStyle{ndims(arraytype)}()
end

# Recover the output/input split shared by all operator operands of `bc`,
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

function broadcast_operator_output_input(bc::Broadcasted)
    ops = operator_operands(bc)
    op1 = first(ops)
    out1 = outputnames(op1)
    inp1 = inputnames(op1)
    for op in Base.tail(ops)
        (issetequal(outputnames(op), out1) && issetequal(inputnames(op), inp1)) ||
            throw(
            ArgumentError(
                "Operator operands disagree on their output/input split: " *
                    "$((out1, inp1)) vs $((outputnames(op), inputnames(op))). " *
                    "Broadcasting operators requires a matching split."
            )
        )
    end
    return out1, inp1
end

function Base.copy(bc::Broadcasted{<:NamedTensorOperatorStyle})
    out, inp = broadcast_operator_output_input(bc)
    result = copy(statebroadcasted(bc))
    return operator(result, out, inp)
end

for f in MATRIX_FUNCTIONS
    @eval begin
        function Base.$f(a::NamedTensorOperator)
            out = outputnames(a)
            inp = inputnames(a)
            return operator($f(state(a), out, inp), out, inp)
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
`conj(x)` with its input dimension names replaced by the corresponding
output names of `a`. `x` carries `a`'s input dimension names and a
fresh trailing rank name. The output and input partition is taken from
`outputnames(a)` and `inputnames(a)`.

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
output and input partition is taken from `outputnames(a)` and
`inputnames(a)`.

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
            return MA.$f(state(a), outputnames(a), inputnames(a); kwargs...)
        end
    end
end

function MAK.project_hermitian(a::NamedTensorOperator; kwargs...)
    h = MAK.project_hermitian(state(a), codomainnames(a), domainnames(a); kwargs...)
    return operator(h, codomainnames(a), domainnames(a))
end
for f in (:sqrth_safe, :invsqrth_safe)
    @eval function MA.$f(a::NamedTensorOperator; kwargs...)
        x = MA.$f(state(a), codomainnames(a), domainnames(a); kwargs...)
        return operator(x, codomainnames(a), domainnames(a))
    end
end
function MA.sqrth_invsqrth_safe(a::NamedTensorOperator; kwargs...)
    x, y = MA.sqrth_invsqrth_safe(state(a), codomainnames(a), domainnames(a); kwargs...)
    return operator(x, codomainnames(a), domainnames(a)),
        operator(y, codomainnames(a), domainnames(a))
end

"""
    Base.one(op::NamedTensorOperator) -> Id

Return the identity operator with the same output/input names and shape as
`op`. `op` is treated as a shape prototype and is not mutated.

The identity acts as the multiplicative identity for `ITensorBase.apply`: it
contracts on the input names and renames the resulting output names back to
the input names, leaving the input unchanged.

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
    out, inp = outputnames(op), inputnames(op)
    return operator(one(state(op), out, inp), out, inp)
end

"""
    LinearAlgebra.tr(op::NamedTensorOperator) -> scalar

Trace of a named operator: contracts each output index with its paired input index and
sums the diagonal, over the operator's intrinsic output/input split.
"""
function LA.tr(op::NamedTensorOperator)
    a = state(op)
    byname = Dict(name(i) => i for i in inds(a))
    output = map(n -> byname[n], Tuple(outputnames(op)))
    input = map(n -> byname[n], Tuple(inputnames(op)))
    return LA.tr(a, output, input)
end

# === similar_operator ===
#
# Allocate an operator with the user-supplied axes as the input. The output
# shares the input direction and either takes explicitly-supplied names or fresh
# `uniquename` outputs. The 5-arg form is canonical, the others fill in defaults.
# The bra/ket flip on the storage side is handled inside `TA.similar_map`.

"""
    similar_operator(prototype, [T,] unnamed_input_axes, [outputnames,] inputnames) -> op
    similar_operator(prototype, [T,] named_input_axes) -> op

Allocate an operator-shaped named array with undefined data, with the
user-supplied side as the input and a matching output.
Element type defaults to `eltype(prototype)`. Output names default to fresh
`uniquename`-generated names. The first form takes unnamed (raw) axes and
explicit names, the second takes already-named axes and reuses their names as
the input. Storage layout (including the bra/ket flip on the input side for
graded axes) is delegated to `TensorAlgebra.similar_map`.

# Examples

```jldoctest
julia> op = similar_operator(zeros(2, 2), (Base.OneTo(2),), (:i,), (:j,));

julia> collect(inputnames(op))
1-element Vector{Symbol}:
 :j
```

See also [`operator`](@ref), [`uniquename`](@ref).
"""
function similar_operator(
        prototype, ::Type{T}, unnamed_input_axes, outputnames, inputnames
    ) where {T}
    output_axes = named.(unnamed_input_axes, outputnames)
    input_axes = named.(unnamed_input_axes, inputnames)
    raw = TA.similar_map(prototype, T, output_axes, input_axes)
    return operator(raw, outputnames, inputnames)
end
function similar_operator(
        prototype, ::Type{T}, unnamed_input_axes, inputnames
    ) where {T}
    outputnames = uniquename.(inputnames)
    return similar_operator(
        prototype, T, unnamed_input_axes, outputnames, inputnames
    )
end
function similar_operator(prototype, ::Type{T}, named_input_axes) where {T}
    return similar_operator(
        prototype, T, unnamed.(named_input_axes), name.(named_input_axes)
    )
end
function similar_operator(prototype, unnamed_input_axes, outputnames, inputnames)
    return similar_operator(
        prototype, eltype(prototype), unnamed_input_axes, outputnames, inputnames
    )
end
function similar_operator(prototype, unnamed_input_axes, inputnames)
    return similar_operator(prototype, eltype(prototype), unnamed_input_axes, inputnames)
end
function similar_operator(prototype, named_input_axes)
    return similar_operator(prototype, eltype(prototype), named_input_axes)
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
