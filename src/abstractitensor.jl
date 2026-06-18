abstract type AbstractITensor <: AbstractNamedDimsArray{Any, Any} end

nameddimsconstructor(::Type{<:IndexName}) = ITensor
dimnametype(::Type{<:AbstractITensor}) = IndexName

Base.ndims(::Type{<:AbstractITensor}) = Any

struct ITensor <: AbstractITensor
    denamed::AbstractArray
    dimnames::Any
    function ITensor(denamed::AbstractArray, dimnames)
        ndims(denamed) == length(dimnames) ||
            throw(ArgumentError("Number of named dims must match ndims."))
        all(dimname -> dimname isa IndexName, dimnames) ||
            throw(ArgumentError("All dimnames must be of type IndexName."))
        return new(denamed, dimnames)
    end
end
dimnames(a::ITensor) = LittleSet(a.dimnames)
denamed(a::ITensor) = a.denamed
Base.parent(a::ITensor) = denamed(a)
