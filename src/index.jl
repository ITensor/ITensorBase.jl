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

tagsstring(tags::Dict{String,String}) = string(tags)

@kwdef struct IndexName <: AbstractName
  id::UInt64 = rand(UInt64)
  tags::Dict{String,String} = Dict{String,String}()
  plev::Int = 0
end
NamedDimsArrays.randname(n::IndexName) = IndexName(; tags=tags(n), plev=plev(n))

id(n::IndexName) = n.id
tags(n::IndexName) = n.tags
plev(n::IndexName) = n.plev

settags(n::IndexName, tags) = @set n.tags = tags

hastag(n::IndexName, tagname::String) = haskey(tags(n), tagname)

gettag(n::IndexName, tagname::String) = tags(n)[tagname]
gettag(n::IndexName, tagname::String, default) = get(tags(n), tagname, default)
function settag(n::IndexName, tagname::String, tag::String)
  newtags = copy(tags(n))
  newtags[tagname] = tag
  return settags(n, newtags)
end
function unsettag(n::IndexName, tagname::String)
  newtags = copy(tags(n))
  delete!(newtags, tagname)
  return settags(n, newtags)
end

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

function Index(r::AbstractUnitRange; kwargs...)
  return Index(r, IndexName(; kwargs...))
end

function Index(length::Int; kwargs...)
  return Index(Base.OneTo(length); kwargs...)
end

# TODO: Define for `NamedDimsArrays.NamedViewIndex`.
id(i::Index) = id(name(i))
tags(i::Index) = tags(name(i))
plev(i::Index) = plev(name(i))

# TODO: Define for `NamedDimsArrays.NamedViewIndex`.
hastag(i::Index, tagname::String) = hastag(name(i), tagname)

# TODO: Define for `NamedDimsArrays.NamedViewIndex`.
gettag(i::Index, tagname::String) = gettag(name(i), tagname)
gettag(i::Index, tagname::String, default) = gettag(name(i), tagname, default)
settag(i::Index, tagname::String, tag::String) = setname(i, settag(name(i), tagname, tag))
unsettag(i::Index, tagname::String) = setname(i, unsettag(name(i), tagname))

setprime(i::Index, plev) = setname(i, setprime(name(i), plev))
prime(i::Index) = setname(i, prime(name(i)))
noprime(i::Index) = setname(i, noprime(name(i)))
sim(i::Index) = setname(i, sim(name(i)))

# TODO: Delete this definition?
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
