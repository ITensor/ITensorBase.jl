using TensorAlgebra: TensorAlgebra

"""
    NamedTensor(array::AbstractArray, dims)

A tensor whose dimensions are labeled by names instead of ordered by position. It pairs
an underlying `array` with one name per dimension (`dims`), so contraction, addition, and
indexing line dimensions up by name. A `NamedTensor` is usually built by calling `randn`, `zeros`,
and the like on indices, or through [`nameddims`](@ref), rather than constructed directly.
[`ITensor`](@ref) is the `NamedTensor` with dimension names that are [`IndexName`](@ref)s.

# Examples

```jldoctest
julia> NamedTensor(zeros(2, 3), (:i, :j))
named(Base.OneTo(2), :i)×named(Base.OneTo(3), :j) NamedTensor{Symbol}:
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
    dimnames::Vector{DimName}
    # The sole inner constructor: enforces the representation invariants (one name per dimension,
    # names distinct) on already-collected names. The outer constructors below normalize the
    # inputs (strip index names, fix the eltype) and funnel through here.
    global function _NamedTensor(unnamed, dimnames::Vector{DimName}) where {DimName}
        TensorAlgebra.ndims(unnamed) == length(dimnames) ||
            throw(ArgumentError("Number of named dims must match ndims."))
        allunique(dimnames) ||
            throw(ArgumentError("Dimension names must be distinct, got $(dimnames)."))
        return new{DimName}(unnamed, dimnames)
    end
end

function NamedTensor{DimName}(unnamed, dimnames) where {DimName}
    # A bare index (`NamedUnitRange` such as `Index`) is ambiguous as `dimnames` (it is itself
    # an iterable of its range values), so require a tuple/vector of names or indices.
    dimnames isa NamedUnitRange && throw(
        ArgumentError(
            "Got a single index (`NamedUnitRange` such as `Index`) as the dimension names; \
            pass a tuple or vector, e.g. `ITensor(array, (i, j))`."
        )
    )
    # `dimnames` can hold plain names or indices; `name` maps an index to its name and is the
    # identity on a plain name. An index's space is ignored, since the array carries the axes.
    return _NamedTensor(unnamed, collect(DimName, name.(dimnames)))
end

# The dimension-name type is inferred from the names, so that indices (`NamedUnitRange`s such as
# `Index`) infer `IndexName` rather than the index type. `name` is the identity on plain names.
function NamedTensor(unnamed, dimnames)
    dimnames′ = name.(dimnames)
    return NamedTensor{eltype(dimnames′)}(unnamed, dimnames′)
end
NamedTensor(a::AbstractNamedTensor, inds) = throw(ArgumentError("Already named."))
NamedTensor(a::AbstractNamedTensor) = NamedTensor(unnamed(a), dimnames(a))

# Minimal interface. The dimnames are stored as (and returned as) a `Vector`.
dimnames(a::NamedTensor) = a.dimnames
unnamed(a::NamedTensor) = a.unnamed
Base.parent(a::NamedTensor) = unnamed(a)

dimnametype(::Type{<:NamedTensor{DimName}}) where {DimName} = DimName

# The parent array is erased at the field level, so its concrete type is not part
# of `NamedTensor`'s signature. An instance still carries the parent, so the instance
# methods recover the concrete type while the type methods report `AbstractArray`.
unnamedtype(a::NamedTensor) = typeof(unnamed(a))
unnamedtype(::Type{<:NamedTensor}) = AbstractArray
parenttype(a::NamedTensor) = typeof(parent(a))
parenttype(::Type{<:NamedTensor}) = AbstractArray
