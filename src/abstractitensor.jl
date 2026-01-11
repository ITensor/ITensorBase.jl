using NamedDimsArrays: NamedDimsArrays, AbstractNamedDimsArray, NamedDimsArray, denamed,
    dimnames, inds, mapinds

abstract type AbstractITensor <: AbstractNamedDimsArray{Any, Any} end

NamedDimsArrays.nameddimsconstructor(::Type{<:IndexName}) = ITensor

Base.ndims(::Type{<:AbstractITensor}) = Any

struct ITensor <: AbstractITensor
    parent::AbstractArray
    inds
    function ITensor(parent::AbstractArray, dims)
        # This checks the shapes of the inputs.
        inds = NamedDimsArrays.to_inds(parent, dims)
        return new(parent, inds)
    end
end
Base.parent(a::ITensor) = getfield(a, :parent)
NamedDimsArrays.inds(a::ITensor) = getfield(a, :inds)
NamedDimsArrays.denamed(a::ITensor) = parent(a)

function ITensor(parent::AbstractArray, i1::Index, i_rest::Index...)
    return ITensor(parent, (i1, i_rest...))
end
function ITensor(parent::AbstractArray)
    return ITensor(parent, ())
end
