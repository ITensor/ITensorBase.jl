using TypeParameterAccessors: TypeParameterAccessors, parenttype

# TODO: Check `allunique(dimnames)`?
struct ITensor{DimName} <: AbstractITensor{DimName}
    denamed::AbstractArray
    dimnames::Vector{DimName}
end

# TODO: Check `allunique(dimnames)`?
function ITensor(denamed::AbstractArray, dims)
    ndims(denamed) == length(dims) ||
        throw(ArgumentError("Number of named dims must match ndims."))
    dimnames = collect(dims)
    return ITensor{eltype(dimnames)}(denamed, dimnames)
end
ITensor(a::AbstractITensor, inds) = throw(ArgumentError("Already named."))
ITensor(a::AbstractITensor) = ITensor(denamed(a), dimnames(a))

# Minimal interface. The dimnames are stored as a `Vector{DimName}`, but the
# accessor still returns a `LittleSet` over a `Tuple` (unchanged public behavior).
dimnames(a::ITensor) = LittleSet(Tuple(a.dimnames))
denamed(a::ITensor) = a.denamed
Base.parent(a::ITensor) = denamed(a)

dimnametype(::Type{<:ITensor{DimName}}) where {DimName} = DimName

# The parent array is erased at the field level, so its concrete type is not part
# of `ITensor`'s signature. An instance still carries the parent, so the instance
# methods recover the concrete type while the type methods report `AbstractArray`.
denamedtype(a::ITensor) = typeof(denamed(a))
denamedtype(::Type{<:ITensor}) = AbstractArray
TypeParameterAccessors.parenttype(a::ITensor) = typeof(parent(a))
TypeParameterAccessors.parenttype(::Type{<:ITensor}) = AbstractArray
