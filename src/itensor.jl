struct ITensor{DimName} <: AbstractITensor{DimName}
    unnamed::AbstractArray
    dimnames::Vector{DimName}
    function ITensor{DimName}(unnamed::AbstractArray, dimnames) where {DimName}
        dimnames = collect(DimName, dimnames)
        ndims(unnamed) == length(dimnames) ||
            throw(ArgumentError("Number of named dims must match ndims."))
        allunique(dimnames) ||
            throw(ArgumentError("Dimension names must be distinct, got $(dimnames)."))
        return new{DimName}(unnamed, dimnames)
    end
end

ITensor(unnamed::AbstractArray, dims) = ITensor{eltype(dims)}(unnamed, dims)
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
