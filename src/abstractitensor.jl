using NamedDimsArrays: NamedDimsArrays, AbstractNamedDimsArray, LittleSet, NamedDimsArray,
    denamed, dimnames, inds, mapinds

abstract type AbstractITensor <: AbstractNamedDimsArray{Any, Any} end

NamedDimsArrays.nameddimsconstructor(::Type{<:IndexName}) = ITensor

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
NamedDimsArrays.dimnames(a::ITensor) = LittleSet(a.dimnames)
NamedDimsArrays.denamed(a::ITensor) = a.denamed
Base.parent(a::ITensor) = denamed(a)
