module ITensorBaseBlockArraysExt
using ArrayLayouts: ArrayLayouts
using BlockArrays: Block, BlockRange
using ITensorBase: AbstractITensor, AbstractNamedInteger, AbstractNamedUnitRange,
    getindex_named, view_nameddims

# The first parameter of `AbstractNamedUnitRange` is its element type, an
# `AbstractNamedInteger`, which is itself no longer an `Integer` subtype. These
# methods disambiguate named-range block indexing from `BlockArrays`' generic
# `AbstractArray` block-indexing methods.
function Base.getindex(r::AbstractNamedUnitRange{<:AbstractNamedInteger}, I::Block{1})
    # TODO: Use `Derive.@interface NamedArrayInterface() r[I]` instead.
    return getindex_named(r, I)
end

function Base.getindex(r::AbstractNamedUnitRange{<:AbstractNamedInteger}, I::BlockRange{1})
    # TODO: Use `Derive.@interface NamedArrayInterface() r[I]` instead.
    return getindex_named(r, I)
end

const BlockIndex{N} = Union{Block{N}, BlockRange{N}, AbstractVector{<:Block{N}}}

function Base.view(a::AbstractITensor, I1::Block{1}, Irest::BlockIndex{1}...)
    # TODO: Use `Derive.@interface ITensorInterface() r[I]` instead.
    return view_nameddims(a, I1, Irest...)
end

function Base.view(a::AbstractITensor, I::Block)
    # TODO: Use `Derive.@interface ITensorInterface() r[I]` instead.
    return view_nameddims(a, Tuple(I)...)
end

function Base.view(a::AbstractITensor, I1::BlockIndex{1}, Irest::BlockIndex{1}...)
    # TODO: Use `Derive.@interface ITensorInterface() r[I]` instead.
    return view_nameddims(a, I1, Irest...)
end

# Fix ambiguity error.
function Base.getindex(
        a::AbstractITensor, I1::BlockRange{1}, Irest::BlockRange{1}...
    )
    return ArrayLayouts.layout_getindex(a, I1, Irest...)
end

# Fix ambiguity errors.
function Base.getindex(a::AbstractITensor, I1::Block{1}, Irest...)
    return copy(view(a, I1, Irest...))
end
function Base.getindex(a::AbstractITensor, I1::AbstractVector, I2::Block{1})
    return copy(view(a, I1, I2))
end
function Base.getindex(a::AbstractITensor, I1::Block{1}, I2::AbstractVector)
    return copy(view(a, I1, I2))
end
function Base.getindex(a::AbstractITensor, I::Block{N}) where {N}
    return copy(view(a, I))
end

end
