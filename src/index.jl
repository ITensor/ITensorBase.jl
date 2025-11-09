using Accessors: @set
using NamedDimsArrays: NamedDimsArrays, AbstractName, AbstractNamedInteger,
    AbstractNamedUnitRange, AbstractNamedVector, dename, name, randname, setname
using Random: Random, AbstractRNG

tagpairstring(pair::Pair) = repr(first(pair)) * "=>" * repr(last(pair))
function tagsstring(tags::Dict{String, String})
    tagpairs = sort(collect(tags); by = first)
    tagpair1, tagpair_rest = Iterators.peel(tagpairs)
    return mapreduce(*, tagpair_rest; init = tagpairstring(tagpair1)) do tagpair
        return "," * tagpairstring(tagpair)
    end
end

struct IndexName <: AbstractName
    id::UInt64
    tags::Dict{String, String}
    plev::Int
end
function IndexName(
        rng::AbstractRNG = Random.default_rng(); id::UInt64 = rand(rng, UInt64),
        tags = Dict{String, String}(), plev::Int = 0,
    )
    return IndexName(id, Dict{String, String}(tags), plev)
end
function NamedDimsArrays.randname(rng::AbstractRNG, n::IndexName)
    return setid(n, rand(rng, UInt64))
end
function NamedDimsArrays.randname(rng::AbstractRNG, ::Type{<:IndexName})
    return IndexName(rng)
end

id(n::IndexName) = getfield(n, :id)
tags(n::IndexName) = getfield(n, :tags)
plev(n::IndexName) = getfield(n, :plev)

using ConstructionBase: getfields
Base.:(==)(n1::IndexName, n2::IndexName) = getfields(n1) == getfields(n2)
Base.isequal(n1::IndexName, n2::IndexName) = isequal(getfields(n1), getfields(n2))
function Base.isless(n1::IndexName, n2::IndexName)
    t1 = (id(n1), plev(n1), keys(tags(n1)), collect(values(tags(n1))))
    t2 = (id(n2), plev(n2), keys(tags(n2)), collect(values(tags(n2))))
    return isless(t1, t2)
end
function Base.hash(n::IndexName, h::UInt)
    h = hash(:IndexName, h)
    h = hash(id(n), h)
    h = hash(tags(n), h)
    h = hash(plev(n), h)
    return h
end

setid(n::IndexName, id) = @set n.id = id
settags(n::IndexName, tags) = @set n.tags = tags
setplev(n::IndexName, plev) = @set n.plev = plev

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

prime(n::IndexName) = setplev(n, plev(n) + 1)
noprime(n::IndexName) = setplev(n, 0)

function Base.show(io::IO, i::IndexName)
    idstr = "id=$(id(i) % 1000)"
    tagsstr = !isempty(tags(i)) ? "|$(tagsstring(tags(i)))" : ""
    primestr = primestring(plev(i))
    str = "IndexName($(idstr)$(tagsstr))$(primestr)"
    print(io, str)
    return nothing
end

struct IndexVal{Value <: Integer} <: AbstractNamedInteger{Value, IndexName}
    value::Value
    name::IndexName
end

# Interface
NamedDimsArrays.dename(i::IndexVal) = i.value
NamedDimsArrays.name(i::IndexVal) = i.name

# Constructor
NamedDimsArrays.named(i::Integer, name::IndexName) = IndexVal(i, name)

struct Index{
        T, Value <: AbstractUnitRange{T},
    } <: AbstractNamedUnitRange{T, Value, IndexName}
    value::Value
    name::IndexName
end

function Index{T, Value}(
        r::AbstractUnitRange{T}; kwargs...
    ) where {T, Value <: AbstractUnitRange{T}}
    return Index{T, Value}(r, IndexName(; kwargs...))
end
function Index{T}(r::AbstractUnitRange{T}; kwargs...) where {T}
    return Index{T, typeof(r)}(r; kwargs...)
end
function Index(r::AbstractUnitRange; kwargs...)
    return Index{eltype(r)}(r; kwargs...)
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

setplev(i::Index, plev) = setname(i, setplev(name(i), plev))
prime(i::Index) = setname(i, prime(name(i)))
noprime(i::Index) = setname(i, noprime(name(i)))

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
    tagsstr = !isempty(tags(i)) ? "|$(tagsstring(tags(i)))" : ""
    primestr = primestring(plev(i))
    str = "Index($(lenstr)$(idstr)$(tagsstr))$(primestr)"
    print(io, str)
    return nothing
end

struct NoncontiguousIndex{T, Value <: AbstractVector{T}} <:
    AbstractNamedVector{T, Value, IndexName}
    value::Value
    name::IndexName
end

# Interface
# TODO: Overload `Base.parent` instead.
NamedDimsArrays.dename(i::NoncontiguousIndex) = i.value
NamedDimsArrays.name(i::NoncontiguousIndex) = i.name

# Constructor
NamedDimsArrays.named(i::AbstractVector, name::IndexName) = NoncontiguousIndex(i, name)
