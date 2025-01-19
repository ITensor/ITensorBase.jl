using Accessors: @set
using NamedDimsArrays:
  NamedDimsArrays,
  AbstractName,
  AbstractNamedInteger,
  AbstractNamedUnitRange,
  AbstractNamedVector,
  dename,
  name,
  randname,
  setname

const Tag = String
const TagSet = Set{Tag}

tagset(tags::String) = Set(filter(!isempty, String.(strip.(split(tags, ",")))))
tagset(tags::TagSet) = tags

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
NamedDimsArrays.randname(n::IndexName) = IndexName(; tags=tags(n), plev=plev(n))

id(n::IndexName) = n.id
tags(n::IndexName) = n.tags
plev(n::IndexName) = n.plev

settags(n::IndexName, tags) = @set n.tags = tags
addtags(n::IndexName, ts) = settags(n, tags(n) ∪ tagset(ts))

setprime(n::IndexName, plev) = @set n.plev = plev
prime(n::IndexName) = setprime(n, plev(n) + 1)
noprime(n::IndexName) = setprime(n, 0)
sim(n::IndexName) = randname(n)

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

function Index(length::Int; tags=TagSet(), kwargs...)
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
noprime(i::Index) = setname(i, noprime(name(i)))
sim(i::Index) = setname(i, sim(name(i)))

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
