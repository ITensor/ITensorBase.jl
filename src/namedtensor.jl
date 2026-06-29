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
    unnamed::AbstractArray
    dimnames::Vector{DimName}
    function NamedTensor{DimName}(unnamed::AbstractArray, dimnames) where {DimName}
        dimnames = collect(DimName, dimnames)
        # Catch the common ITensors.jl-style mistake of passing indices as the names.
        any(dimname -> dimname isa NamedUnitRange, dimnames) && throw(
            ArgumentError(
                "The `NamedTensor` constructor takes dimension names only, not indices \
                (`NamedUnitRange`s), got $(dimnames). To build a `NamedTensor` from an \
                array and indices, index the array instead, as in `array[i, j]`."
            )
        )
        ndims(unnamed) == length(dimnames) ||
            throw(ArgumentError("Number of named dims must match ndims."))
        allunique(dimnames) ||
            throw(ArgumentError("Dimension names must be distinct, got $(dimnames)."))
        return new{DimName}(unnamed, dimnames)
    end
end

NamedTensor(unnamed::AbstractArray, dims) = NamedTensor{eltype(dims)}(unnamed, dims)
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
