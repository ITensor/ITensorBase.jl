module ITensorBase

export ITensor, Index

using Accessors: @set
using MapBroadcast: Mapped
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
  setname,
  setnameddimsindices,
  unname

const Tag = String
const TagSet = Set{Tag}

tagset(tags::String) = Set(filter(!isempty, String.(strip.(split(tags, ",")))))

function tagsstring(tags::TagSet)
  str = ""
  length(tags) == 0 && return str
  tags_vec = collect(tags)
  for n in 1:(length(tags_vec) - 1)
    str *= "$(tags_vec[n]),"
  end
  str *= "$(tags_vec[end])"
  return str
end

@kwdef struct IndexName <: AbstractName
  id::UInt64 = rand(UInt64)
  tags::TagSet = TagSet()
  plev::Int = 0
end
NamedDimsArrays.randname(n::IndexName) = IndexName()

id(n::IndexName) = n.id
tags(n::IndexName) = n.tags
plev(n::IndexName) = n.plev

settags(n::IndexName, tags) = @set n.tags = tags
addtags(n::IndexName, ts) = settags(n, tags(n) ∪ tagset(ts))

setprime(n::IndexName, plev) = @set n.plev = plev
prime(n::IndexName) = setprime(n, plev(n) + 1)

function Base.show(io::IO, i::IndexName)
  idstr = "id=$(id(i) % 1000)"
  tagsstr = !isempty(tags(i)) ? "|\"$(tagsstring(tags(i)))\"" : ""
  primestr = primestring(plev(i))
  str = "IndexName($(idstr)$(tagsstr))$(primestr)"
  print(io, str)
  return nothing
end

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

function Index(length::Int; tags, kwargs...)
  return Index(Base.OneTo(length), IndexName(; tags=tagset(tags), kwargs...))
end
function Index(length::Int, tags::String; kwargs...)
  return Index(Base.OneTo(length), IndexName(; kwargs..., tags=tagset(tags)))
end

# TODO: Define for `NamedDimsArrays.NamedViewIndex`.
id(i::Index) = id(name(i))
tags(i::Index) = tags(name(i))
plev(i::Index) = plev(name(i))

# TODO: Define for `NamedDimsArrays.NamedViewIndex`.
addtags(i::Index, tags) = setname(i, addtags(name(i), tags))
prime(i::Index) = setname(i, prime(name(i)))
Base.adjoint(i::Index) = prime(i)

# Interface
# TODO: Overload `Base.parent` instead.
NamedDimsArrays.dename(i::Index) = i.value
NamedDimsArrays.name(i::Index) = i.name

# Constructor
NamedDimsArrays.named(i::AbstractUnitRange, name::IndexName) = Index(i, name)

function primestring(plev)
  if plev < 0
    return " (warning: prime level $plev is less than 0)"
  end
  if plev == 0
    return ""
  elseif plev > 3
    return "'$plev"
  else
    return "'"^plev
  end
end

function Base.show(io::IO, i::Index)
  lenstr = "length=$(dename(length(i)))"
  idstr = "|id=$(id(i) % 1000)"
  tagsstr = !isempty(tags(i)) ? "|\"$(tagsstring(tags(i)))\"" : ""
  primestr = primestring(plev(i))
  str = "Index($(lenstr)$(idstr)$(tagsstr))$(primestr)"
  print(io, str)
  return nothing
end

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
end
Base.parent(a::ITensor) = a.parent
NamedDimsArrays.nameddimsindices(a::ITensor) = a.nameddimsindices

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

# TODO: Use `replaceinds`/`mapinds`, based on
# `replacenameddimsindices`/`mapnameddimsindices`.
prime(a::AbstractITensor) = setinds(a, prime.(inds(a)))

include("quirks.jl")

end
