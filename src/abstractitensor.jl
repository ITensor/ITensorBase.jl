using MapBroadcast: Mapped
using NamedDimsArrays:
  NamedDimsArrays,
  AbstractNamedDimsArray,
  NamedDimsArray,
  dename,
  dimnames,
  mapnameddimsindices,
  nameddimsindices,
  replacenameddimsindices,
  setnameddimsindices

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
function specify_eltype(a::AbstractArray, elt::Type)
  return a
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

using DerivableInterfaces: @derive, @interface, AbstractArrayInterface

abstract type AbstractAllocatableArrayInterface <: AbstractArrayInterface end
struct AllocatableArrayInterface <: AbstractAllocatableArrayInterface end

unallocatable(a::AbstractITensor) = NamedDimsArray(a)

function setindex_allocatable!(a::AbstractArray, value, I...)
  allocate!(specify_eltype!(a, typeof(value)))
  # TODO: Maybe use `@interface interface(a) a[I...] = value`?
  unallocatable(a)[I...] = value
  return a
end

# TODO: Combine these by using `Base.to_indices`.
@interface ::AbstractAllocatableArrayInterface function Base.setindex!(
  a::AbstractArray, value, I::Int...
)
  setindex_allocatable!(a, value, I...)
  return a
end
@interface ::AbstractAllocatableArrayInterface function Base.setindex!(
  a::AbstractArray, value, I::AbstractNamedInteger...
)
  setindex_allocatable!(a, value, I...)
  return a
end

@derive AllocatableArrayInterface() (T=AbstractITensor,) begin
  Base.setindex!(::T, ::Any, ::Int...)
  Base.setindex!(::T, ::Any, ::AbstractNamedInteger...)
end

mutable struct ITensor <: AbstractITensor
  parent::AbstractArray
  nameddimsindices
  function ITensor(parent::AbstractArray, dims)
    # This checks the shapes of the inputs.
    nameddimsindices = NamedDimsArrays.to_nameddimsindices(parent, dims)
    return new(parent, nameddimsindices)
  end
end
Base.parent(a::ITensor) = getfield(a, :parent)
NamedDimsArrays.nameddimsindices(a::ITensor) = getfield(a, :nameddimsindices)
NamedDimsArrays.dename(a::ITensor) = parent(a)

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
  return ITensor(zeros(elt, length.(dename.(I))...), I)
end

function ITensor(I1::Index, I_rest::Index...)
  I = (I1, I_rest...)
  return ITensor(Zeros{UnspecifiedZero}(length.(dename.(I))...), I)
end

function ITensor()
  return ITensor(Zeros{UnspecifiedZero}(), ())
end

inds(a::AbstractITensor) = nameddimsindices(a)
setinds(a::AbstractITensor, inds) = setnameddimsindices(a, inds)

function uniqueinds(a1::AbstractITensor, a_rest::AbstractITensor...)
  return setdiff(inds(a1), inds.(a_rest)...)
end
function uniqueind(a1::AbstractITensor, a_rest::AbstractITensor...)
  return only(uniqueinds(a1, a_rest...))
end

function commoninds(a1::AbstractITensor, a_rest::AbstractITensor...)
  return intersect(inds(a1), inds.(a_rest)...)
end
function commonind(a1::AbstractITensor, a_rest::AbstractITensor...)
  return only(commoninds(a1, a_rest...))
end

function replaceinds(a::AbstractITensor, replacements::Pair...)
  return replacenameddimsindices(a, replacements...)
end
function replaceinds(f, a::AbstractITensor)
  return replacenameddimsindices(f, a)
end

function mapinds(f, a::AbstractITensor)
  return mapnameddimsindices(f, a)
end

prime(a::AbstractITensor) = replaceinds(prime, a)
noprime(a::AbstractITensor) = replaceinds(noprime, a)
sim(a::AbstractITensor) = replaceinds(sim, a)

using VectorInterface: VectorInterface, scalartype
VectorInterface.scalartype(a::AbstractITensor) = scalartype(unallocatable(a))
