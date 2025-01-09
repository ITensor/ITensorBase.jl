module ITensorBase

using BroadcastMapConversion: Mapped
using NamedDimsArrays:
  NamedDimsArrays,
  AbstractName,
  AbstractNamedDimsArray,
  AbstractNamedInteger,
  AbstractNamedUnitRange,
  AbstractNamedVector,
  NamedDimsArray,
  dename,
  dimnames,
  name,
  named,
  nameddimsindices,
  unname

@kwdef struct IndexName <: AbstractName
  id::UInt64 = rand(UInt64)
  plev::Int = 0
  tags::Set{String} = Set{String}()
  namedtags::Dict{Symbol,String} = Dict{Symbol,String}()
end
NamedDimsArrays.randname(n::IndexName) = IndexName()

struct IndexVal{Value<:Integer} <: AbstractNamedInteger{Value,IndexName}
  value::Value
  name::IndexName
end

# Interface
NamedDimsArrays.dename(i::IndexVal) = i.value
NamedDimsArrays.name(i::IndexVal) = i.name

# Constructor
NamedDimsArrays.named(i::Integer, name::IndexName) = IndexVal(i, name)

struct Index{T,Value<:AbstractUnitRange{T}} <: AbstractNamedUnitRange{T,Value,IndexName}
  value::Value
  name::IndexName
end

Index(length::Int) = Index(Base.OneTo(length), IndexName())

# Interface
# TODO: Overload `Base.parent` instead.
NamedDimsArrays.dename(i::Index) = i.value
NamedDimsArrays.name(i::Index) = i.name

# Constructor
NamedDimsArrays.named(i::AbstractUnitRange, name::IndexName) = Index(i, name)

struct NoncontiguousIndex{T,Value<:AbstractVector{T}} <:
       AbstractNamedVector{T,Value,IndexName}
  value::Value
  name::IndexName
end

# Interface
# TODO: Overload `Base.parent` instead.
NamedDimsArrays.dename(i::NoncontiguousIndex) = i.value
NamedDimsArrays.name(i::NoncontiguousIndex) = i.name

# Constructor
NamedDimsArrays.named(i::AbstractVector, name::IndexName) = NoncontiguousIndex(i, name)

abstract type AbstractITensor <: AbstractNamedDimsArray{Any,Any} end

NamedDimsArrays.nameddimsarraytype(::Type{<:IndexName}) = ITensor

Base.ndims(::Type{<:AbstractITensor}) = Any

using FillArrays: Zeros
using UnallocatedArrays: UnallocatedZeros, allocate
using UnspecifiedTypes: UnspecifiedZero

# TODO: Make this more general, maybe with traits `is_unallocated`
# and `is_eltype_unspecified`.
function specify_eltype(a::Zeros{UnspecifiedZero}, elt::Type)
  return Zeros{elt}(axes(a))
end

# TODO: Use `adapt` to reach down into the storage.
function specify_eltype!(a::AbstractITensor, elt::Type)
  setdenamed!(a, specify_eltype(dename(a), elt))
  return a
end

# Assume it is allocated.
allocate!(a::AbstractArray) = a

# TODO: Use `adapt` to reach down into the storage.
function allocate!(a::AbstractITensor)
  setdenamed!(a, allocate(dename(a)))
  return a
end

using Derive: @derive, @interface, AbstractArrayInterface

abstract type AbstractAllocatableArrayInterface <: AbstractArrayInterface end
struct AllocatableArrayInterface <: AbstractAllocatableArrayInterface end

unallocatable(a::AbstractITensor) = NamedDimsArray(a)

@interface ::AbstractAllocatableArrayInterface function Base.setindex!(
  a::AbstractArray, value, I::Int...
)
  allocate!(specify_eltype!(a, typeof(value)))
  # TODO: Maybe use `@interface interface(a) a[I...] = value`?
  unallocatable(a)[I...] = value
  return a
end

@derive AllocatableArrayInterface() (T=AbstractITensor,) begin
  Base.setindex!(::T, ::Any, ::Int...)
end

mutable struct ITensor <: AbstractITensor
  parent::AbstractArray
  nameddimsindices
end
Base.parent(a::ITensor) = a.parent
NamedDimsArrays.nameddimsindices(a::ITensor) = a.nameddimsindices

using Accessors: @set
setdenamed(a::ITensor, denamed) = (@set a.parent = denamed)
setdenamed!(a::ITensor, denamed) = (a.parent = denamed)

function ITensor(I1::Index, I_rest::Index...)
  I = (I1, I_rest...)
  return ITensor(Zeros{UnspecifiedZero}(length.(dename.(I))...), I)
end

end
