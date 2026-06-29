"""
    ITensor(array::AbstractArray, dimnames)

A dense tensor whose dimensions are labeled by names instead of ordered by position. It pairs
an underlying `array` with one name per dimension (`dimnames`), so contraction, addition, and
indexing line dimensions up by name. An `ITensor` is usually built by calling `randn`, `zeros`,
and the like on indices, or through [`nameddims`](@ref), rather than constructed directly.

# Examples

```jldoctest
julia> ITensor(zeros(2, 3), (:i, :j))
named(Base.OneTo(2), :i)×named(Base.OneTo(3), :j) ITensor{Symbol}:
2×3 Matrix{Float64}:
 0.0  0.0  0.0
 0.0  0.0  0.0
```
"""
struct ITensor{DimName} <: AbstractITensor{DimName}
    unnamed::AbstractArray
    dimnames::Vector{DimName}
    function ITensor{DimName}(unnamed::AbstractArray, dimnames) where {DimName}
        dimnames = collect(DimName, dimnames)
        # Catch the common ITensors.jl-style mistake of passing indices as the names.
        any(dimname -> dimname isa NamedUnitRange, dimnames) && throw(
            ArgumentError(
                "Dimension names must be names, not indices (`NamedUnitRange`s), got \
                $(dimnames). To build an `ITensor` from an array and indices, index the \
                array instead, as in `array[i, j]`."
            )
        )
        ndims(unnamed) == length(dimnames) ||
            throw(ArgumentError("Number of named dims must match ndims."))
        allunique(dimnames) ||
            throw(ArgumentError("Dimension names must be distinct, got $(dimnames)."))
        return new{DimName}(unnamed, dimnames)
    end
end

ITensor(unnamed::AbstractArray, dimnames) = ITensor{eltype(dimnames)}(unnamed, dimnames)
ITensor(a::AbstractITensor, inds) = throw(ArgumentError("Already named."))
ITensor(a::AbstractITensor) = ITensor(unnamed(a), dimnames(a))

# Minimal interface. The dimnames are stored as (and returned as) a `Vector`.
dimnames(a::ITensor) = a.dimnames
unnamed(a::ITensor) = a.unnamed
Base.parent(a::ITensor) = unnamed(a)

dimnametype(::Type{<:ITensor{DimName}}) where {DimName} = DimName

# The parent array is erased at the field level, so its concrete type is not part
# of `ITensor`'s signature. An instance still carries the parent, so the instance
# methods recover the concrete type while the type methods report `AbstractArray`.
unnamedtype(a::ITensor) = typeof(unnamed(a))
unnamedtype(::Type{<:ITensor}) = AbstractArray
parenttype(a::ITensor) = typeof(parent(a))
parenttype(::Type{<:ITensor}) = AbstractArray
