using MapBroadcast: Mapped
using NamedDimsArrays: NamedDimsArrays, AbstractNamedDimsArray, NamedDimsArray, denamed,
    dimnames, inds, mapinds

abstract type AbstractITensor <: AbstractNamedDimsArray{Any, Any} end

NamedDimsArrays.nameddimsconstructor(::Type{<:IndexName}) = ITensor

Base.ndims(::Type{<:AbstractITensor}) = Any

using FillArrays: Zeros
using UnallocatedArrays: UnallocatedZeros, allocate
using UnspecifiedTypes: UnspecifiedZero

# TODO: Make this more general, maybe with traits `is_unallocated`
# and `is_eltype_unspecified`.
function specify_eltype(a::Zeros{UnspecifiedZero}, elt::Type)
    return Zeros{elt}(axes(a))
end
specify_eltype(a::AbstractArray, elt::Type) = a

# TODO: Use `adapt` to reach down into the storage.
function specify_eltype!(a::AbstractITensor, elt::Type)
    setdenamed!(a, specify_eltype(denamed(a), elt))
    return a
end

# Assume it is allocated.
allocate!(a::AbstractArray) = a

# TODO: Use `adapt` to reach down into the storage.
function allocate!(a::AbstractITensor)
    setdenamed!(a, allocate(denamed(a)))
    return a
end

unallocatable(a::AbstractITensor) = NamedDimsArray(a)

function setindex_allocatable!(a::AbstractArray, value, I...)
    allocate!(specify_eltype!(a, typeof(value)))
    # TODO: Maybe use `@interface interface(a) a[I...] = value`?
    unallocatable(a)[I...] = value
    return a
end

# TODO: Combine these by using `Base.to_indices`.
function Base.setindex!(a::AbstractITensor, value, I::Int...)
    setindex_allocatable!(a, value, I...)
    return a
end
function Base.setindex!(a::AbstractITensor, value, I::AbstractNamedInteger...)
    setindex_allocatable!(a, value, I...)
    return a
end

mutable struct ITensor <: AbstractITensor
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

using Accessors: @set
setdenamed(a::ITensor, denamed) = (@set a.parent = denamed)
setdenamed!(a::ITensor, denamed) = (a.parent = denamed)

function ITensor(elt::Type, I1::Index, I_rest::Index...)
    I = (I1, I_rest...)
    # TODO: Use `FillArrays.Zeros`.
    return ITensor(zeros(elt, length.(denamed.(I))...), I)
end

function ITensor(I1::Index, I_rest::Index...)
    I = (I1, I_rest...)
    return ITensor(Zeros{UnspecifiedZero}(length.(denamed.(I))...), I)
end

function ITensor()
    return ITensor(Zeros{UnspecifiedZero}(), ())
end
