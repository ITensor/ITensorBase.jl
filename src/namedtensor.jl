using TensorAlgebra: TensorAlgebra

"""
    NamedTensor(array::AbstractArray, names)

A tensor whose dimensions are labeled by names instead of ordered by position. It pairs
an underlying `array` with one name per dimension (`names`), so contraction, addition, and
indexing line dimensions up by name. A `NamedTensor` is usually built by calling `randn`, `zeros`,
and the like on indices, or by indexing an array by name, rather than constructed directly.
[`ITensor`](@ref) is the `NamedTensor` with dimension names that are [`IndexName`](@ref)s.

A dimension is given either as a plain name or as an index (a [`NamedUnitRange`](@ref) such as
an [`Index`](@ref)). An index also asserts a space, which has to match the array's corresponding
axis, duality included, and an `ArgumentError` is thrown if it does not. A plain name asserts
nothing, so the array's axis stands.

See also the `NamedTensor(unnamed, codomain_names, domain_names)` method for the map-shaped
form.

# Examples

```jldoctest
julia> NamedTensor(zeros(2, 3), (:i, :j))
NamedOneTo(2, :i)×NamedOneTo(3, :j) NamedTensor{Symbol}:
2×3 Matrix{Float64}:
 0.0  0.0  0.0
 0.0  0.0  0.0
```
"""
struct NamedTensor{DimName} <: AbstractNamedTensor{DimName}
    # The parent is usually an `AbstractArray`, but the field is left untyped so a non-array
    # tensor backend (e.g. a TensorKit `TensorMap`, reached through TensorAlgebra's `ndims`/
    # `axes`/algebra interface) can be the parent directly. See the TensorKit extension.
    unnamed::Any
    names::Vector{DimName}
    # The sole inner constructor: enforces the representation invariants (one name per dimension,
    # names distinct) on already-collected names. The outer constructors below normalize the
    # inputs (strip index names, fix the eltype) and funnel through here.
    global function _NamedTensor(unnamed, names::Vector{DimName}) where {DimName}
        TensorAlgebra.ndims(unnamed) == length(names) ||
            throw(ArgumentError("Number of named dims must match ndims."))
        allunique(names) ||
            throw(ArgumentError("Dimension names must be distinct, got $(names)."))
        return new{DimName}(unnamed, names)
    end
end

# A dimension given as an index asserts a space, so it has to agree with the array's axis; a
# bare name asserts nothing, so only the index case is checked. The comparison is on the
# underlying ranges (`space`) rather than on the indices, because `==` on an `Index` ignores
# duality and would pass a dual/non-dual mismatch.
function checkspaces(unnamed, names)
    # A count mismatch is the inner constructor's error to report, so skip rather than compare
    # against a padded axis.
    length(names) == TensorAlgebra.ndims(unnamed) || return nothing
    for (d, n) in enumerate(names)
        checkspace(unnamed, d, n, identity)
    end
    return nothing
end

# Codomain/domain form: the domain names are given codomain-facing while the storage holds them
# dualized (the convention of `TensorAlgebra.similar_map` and `TensorAlgebra.unmatricize`), so a
# domain index asserts the dual of its own space.
function checkspaces(unnamed, codomain_names, domain_names)
    ncodomain = length(codomain_names)
    # A count mismatch is the inner constructor's error to report, so skip rather than compare
    # against a padded axis.
    ncodomain + length(domain_names) == TensorAlgebra.ndims(unnamed) || return nothing
    for (d, n) in enumerate(codomain_names)
        checkspace(unnamed, d, n, identity)
    end
    for (d, n) in enumerate(domain_names)
        checkspace(unnamed, ncodomain + d, n, conj)
    end
    return nothing
end

# `dualize` maps a dimension's space to the space the storage holds at that position (`identity`
# in the codomain, `conj` in the domain). It is taken as a function rather than as a precomputed
# space because `space` is only defined once `n` is known to be an index.
function checkspace(unnamed, d, n, dualize)
    n isa NamedUnitRange || return nothing
    expected = dualize(space(n))
    ax = TensorAlgebra.axes(unnamed, d)
    ax == expected && return nothing
    asserted = if dualize === identity
        "whose space $(expected)"
    else
        "whose space dualized for its domain position, $(expected),"
    end
    throw(
        ArgumentError(
            "Dimension $(d) was given the index $(n), $(asserted) does not match the \
            corresponding axis $(ax) of the array."
        )
    )
end

# A lone index is ambiguous as a group of dimensions (a `NamedUnitRange` is itself an iterable
# of its range values), so it is rejected rather than splatted into its elements.
function checknotindex(names)
    names isa NamedUnitRange && throw(
        ArgumentError(
            "Got a single index (`NamedUnitRange` such as `Index`) as the dimension names. \
            Pass a tuple or vector, e.g. `ITensor(array, (i, j))`."
        )
    )
    return nothing
end

# `names` can hold plain names or indices (`NamedUnitRange`s such as `Index`): `name` maps an
# index to its name and is the identity on a plain name, so only an index's name is stored (the
# array carries the axes), after `checkspaces` has checked that the space it asserts agrees.
# The methods below repeat this normalization rather than delegating to one another, so each
# strips names exactly once.
function NamedTensor{DimName}(unnamed, names) where {DimName}
    checknotindex(names)
    checkspaces(unnamed, names)
    return _NamedTensor(unnamed, collect(DimName, name.(names)))
end
# The dimension-name type is inferred from the names, so indices infer `IndexName`, not their type.
function NamedTensor(unnamed, names)
    checknotindex(names)
    checkspaces(unnamed, names)
    return _NamedTensor(unnamed, collect(name.(names)))
end

"""
    NamedTensor(unnamed, codomain_names, domain_names)

A tensor whose dimensions are split into a codomain group and a domain group, as a map from
the domain to the codomain. The storage holds the codomain dimensions first and the domain
dimensions last. `codomain_names` and `domain_names` hold names or indices, and the domain
indices are given codomain-facing: an index `n` in `domain_names` asserts that the storage's
axis is the dual `conj(space(n))`, matching how `TensorAlgebra.similar_map` and
`TensorAlgebra.unmatricize` build map-shaped storage.

# Examples

```jldoctest
julia> i, j = NamedUnitRange(1:2, :i), NamedUnitRange(1:3, :j);

julia> NamedTensor(zeros(2, 3), (i,), (j,))
NamedOneTo(2, :i)×NamedOneTo(3, :j) NamedTensor{Symbol}:
2×3 Matrix{Float64}:
 0.0  0.0  0.0
 0.0  0.0  0.0
```
"""
function NamedTensor(unnamed, codomain_names, domain_names)
    checknotindex(codomain_names)
    checknotindex(domain_names)
    checkspaces(unnamed, codomain_names, domain_names)
    return _NamedTensor(
        unnamed, collect((name.(codomain_names)..., name.(domain_names)...))
    )
end
function NamedTensor{DimName}(unnamed, codomain_names, domain_names) where {DimName}
    checknotindex(codomain_names)
    checknotindex(domain_names)
    checkspaces(unnamed, codomain_names, domain_names)
    return _NamedTensor(
        unnamed, collect(DimName, (name.(codomain_names)..., name.(domain_names)...))
    )
end
NamedTensor(a::AbstractNamedTensor, inds) = throw(ArgumentError("Already named."))
NamedTensor(a::AbstractNamedTensor) = NamedTensor(unnamed(a), names(a))

# Minimal interface. The names are stored as (and returned as) a `Vector`.
names(a::NamedTensor) = a.names
unnamed(a::NamedTensor) = a.unnamed
Base.parent(a::NamedTensor) = unnamed(a)

# The parent array is erased at the field level, so its concrete type is not part
# of `NamedTensor`'s signature. An instance still carries the parent, so the instance
# methods recover the concrete type while the type methods report `AbstractArray`.
unnamedtype(a::NamedTensor) = typeof(unnamed(a))
unnamedtype(::Type{<:NamedTensor}) = AbstractArray
parenttype(a::NamedTensor) = typeof(parent(a))
parenttype(::Type{<:NamedTensor}) = AbstractArray
