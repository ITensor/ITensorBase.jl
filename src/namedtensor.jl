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

See also the `NamedTensor(unnamed, codomain_inds, domain_inds)` method for the map-shaped
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
    # The sole inner constructor, and unchecked: it takes already-collected names and wraps them.
    # Validating the arguments is `TensorAlgebra.check_input`'s job, which every outer constructor
    # below calls before normalizing the inputs (stripping index names, fixing the eltype) and
    # funnelling through here, so the whole contract reads in one place.
    global function _NamedTensor(unnamed, names::Vector{DimName}) where {DimName}
        return new{DimName}(unnamed, names)
    end
end

# A dimension given as an index asserts a space, so it has to agree with the corresponding axis;
# a bare name asserts nothing, so only the index case is checked. The comparison is on the
# underlying range (`space`) rather than on the index, because `==` on an `Index` ignores duality
# and would pass a dual/non-dual mismatch.
checkspace(ax, n) = nothing
function checkspace(ax, n::NamedUnitRange)
    ax == space(n) && return nothing
    throw(
        ArgumentError(
            "The index $(n) asserts the space $(space(n)), which does not match the \
            corresponding axis $(ax) of the array."
        )
    )
end

# One name per dimension, and no name used twice. `name` is the identity on a plain name, so
# these compare what actually gets stored: an index and its own bare name collide.
function checkndims(unnamed, nnames)
    TensorAlgebra.ndims(unnamed) == nnames ||
        throw(ArgumentError("Number of named dims must match ndims."))
    return nothing
end
function checkdistinct(names)
    allunique(Iterators.map(name, names)) || throw(
        ArgumentError(
            "Dimension names must be distinct, got $(collect(Iterators.map(name, names)))."
        )
    )
    return nothing
end

# A `NamedUnitRange` is itself an iterable of its range values, so a lone index passed where the
# dimension names were expected would splat into integers rather than name one dimension.
checknotind(names) = nothing
function checknotind(names::NamedUnitRange)
    throw(
        ArgumentError(
            "Got a single index (`NamedUnitRange` such as `Index`) as the dimension names. \
            Pass a tuple or vector, e.g. `ITensor(array, (i, j))`."
        )
    )
end

# `TensorAlgebra.ndims_codomain` defaults to `ndims`, so a plain `Array` reports all-codomain and
# an arity assertion on its own would reject the ordinary dense case. `has_bipartition` is what
# separates a bipartition the storage genuinely carries from that default, so the claimed one is
# only checkable against storage that says it has one.
function checkbipartition(unnamed, codomain_inds, domain_inds)
    TensorAlgebra.has_bipartition(unnamed) || return nothing
    ncodomain = TensorAlgebra.ndims_codomain(unnamed)
    ncodomain == length(codomain_inds) && return nothing
    throw(
        ArgumentError(
            "Got $(length(codomain_inds)) codomain and $(length(domain_inds)) domain \
            dimensions, but the array is a map from $(TensorAlgebra.ndims_domain(unnamed)) \
            dimensions to $(ncodomain)."
        )
    )
end

# Each dimension is checked against the axis the storage holds at its position. `checkndims` has
# already established that the counts agree, so these walk every dimension.
function checkspaces(unnamed, names)
    foreach(checkspace, TensorAlgebra.axes(unnamed), names)
    return nothing
end
# The storage holds the domain axes dualized (the convention of `TensorAlgebra.similar_map` and
# `TensorAlgebra.unmatricize`) while the domain inds are given codomain-facing, so `conj` puts
# that half back in the form the inds are written in. It is a no-op on a dense axis.
function checkspaces(unnamed, codomain_inds, domain_inds)
    ncodomain = length(codomain_inds)
    axes = TensorAlgebra.axes(unnamed)
    foreach(checkspace, Iterators.take(axes, ncodomain), codomain_inds)
    foreach(checkspace, Iterators.map(conj, Iterators.drop(axes, ncodomain)), domain_inds)
    return nothing
end

# The constructors' input check, keyed on the constructor the way TensorAlgebra keys its other
# validation hooks (`check_input(unmatricize, m, axes_codomain, axes_domain)`), and taking the
# constructor's own arguments.
function TensorAlgebra.check_input(::Type{<:NamedTensor}, unnamed, names)
    checknotind(names)
    checkndims(unnamed, length(names))
    checkdistinct(names)
    checkspaces(unnamed, names)
    return nothing
end
function TensorAlgebra.check_input(
        ::Type{<:NamedTensor},
        unnamed,
        codomain_inds,
        domain_inds
    )
    checknotind(codomain_inds)
    checknotind(domain_inds)
    checkndims(unnamed, length(codomain_inds) + length(domain_inds))
    checkdistinct(Iterators.flatten((codomain_inds, domain_inds)))
    checkbipartition(unnamed, codomain_inds, domain_inds)
    checkspaces(unnamed, codomain_inds, domain_inds)
    return nothing
end

# `names` can hold plain names or indices (`NamedUnitRange`s such as `Index`): `name` maps an
# index to its name and is the identity on a plain name, so only an index's name is stored (the
# array carries the axes), after `check_input` has checked that the space it asserts agrees.
function NamedTensor{DimName}(unnamed, names) where {DimName}
    TensorAlgebra.check_input(NamedTensor, unnamed, names)
    return _NamedTensor(unnamed, collect(DimName, name.(names)))
end
# The dimension-name type is inferred from the names, so indices infer `IndexName`, not their type.
function NamedTensor(unnamed, names)
    TensorAlgebra.check_input(NamedTensor, unnamed, names)
    return _NamedTensor(unnamed, collect(name.(names)))
end

"""
    NamedTensor(unnamed, codomain_inds, domain_inds)

A tensor whose dimensions are split into a codomain group and a domain group, as a map from
the domain to the codomain. The storage holds the codomain dimensions first and the domain
dimensions last. `codomain_inds` and `domain_inds` hold indices or plain names, and the domain
indices are given codomain-facing: the storage holds the domain axes dualized, following
`TensorAlgebra.similar_map` and `TensorAlgebra.unmatricize`, so an index in `domain_inds` asserts
the undualized space.

When the array carries a bipartition of its own, the claimed one has to agree with it. Dense
storage carries none, so any bipartition may be claimed over it.

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
function NamedTensor(unnamed, codomain_inds, domain_inds)
    TensorAlgebra.check_input(NamedTensor, unnamed, codomain_inds, domain_inds)
    return _NamedTensor(
        unnamed,
        collect((name.(codomain_inds)..., name.(domain_inds)...))
    )
end
function NamedTensor{DimName}(unnamed, codomain_inds, domain_inds) where {DimName}
    TensorAlgebra.check_input(NamedTensor, unnamed, codomain_inds, domain_inds)
    return _NamedTensor(
        unnamed, collect(DimName, (name.(codomain_inds)..., name.(domain_inds)...))
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
