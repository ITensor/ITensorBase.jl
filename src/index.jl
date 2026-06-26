using Accessors: @set
using Random: AbstractRNG, RandomDevice
using UUIDs: UUID, uuid4

tagpairstring(pair::Pair) = repr(first(pair)) * "=>" * repr(last(pair))
function tagsstring(tags::Dict{String, String})
    tagpairs = sort(collect(tags); by = first)
    tagpair1, tagpair_rest = Iterators.peel(tagpairs)
    return mapreduce(*, tagpair_rest; init = tagpairstring(tagpair1)) do tagpair
        return "," * tagpairstring(tagpair)
    end
end

struct IndexName <: AbstractName
    id::UUID
    tags::Dict{String, String}
    plev::Int
end
function IndexName(
        rng::AbstractRNG = RandomDevice(); id::UUID = uuid4(rng),
        tags = Dict{String, String}(), plev::Int = 0
    )
    return IndexName(id, Dict{String, String}(tags), plev)
end
function uniquename(rng::AbstractRNG, n::IndexName)
    return setid(n, uuid4(rng))
end
function uniquename(rng::AbstractRNG, ::Type{<:IndexName})
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

"""
    prime(i)

Increment the prime level of an index or index name by one, returning a new index that
is distinct from `i`. Priming is the usual way to make a second copy of an index that
carries the same tags but is not contracted against the original. The inverse is
[`noprime`](@ref), which resets the prime level to zero.

# Examples

```jldoctest
julia> i = Index(2);

julia> prime(i) == i
false

julia> noprime(prime(i)) == i
true
```

See also [`noprime`](@ref), [`Index`](@ref).
"""
function prime end

"""
    noprime(i)

Reset the prime level of an index or index name to zero, returning a new index. This
undoes any number of [`prime`](@ref) calls.

# Examples

```jldoctest
julia> i = Index(2);

julia> noprime(prime(i)) == i
true
```

See also [`prime`](@ref), [`Index`](@ref).
"""
function noprime end

prime(n::IndexName) = setplev(n, plev(n) + 1)
noprime(n::IndexName) = setplev(n, 0)

# Show a short prefix of the `UUID` id rather than the full 36-character string,
# enough to disambiguate indices at a glance without dominating the output. A
# leading prefix (here the first hyphen-delimited group) is the usual short-id
# convention, as in git short hashes and Docker short ids.
shortid(id::UUID) = first(string(id), 8)

function Base.show(io::IO, i::IndexName)
    idstr = "id=$(shortid(id(i)))"
    tagsstr = !isempty(tags(i)) ? "|$(tagsstring(tags(i)))" : ""
    primestr = primestring(plev(i))
    str = "IndexName($(idstr)$(tagsstr))$(primestr)"
    print(io, str)
    return nothing
end

"""
    Index(length)
    Index(range)

An index of an `ITensor`: a named unit range whose name is a freshly minted, unique
identifier carrying tags and a prime level. `Index(2)` makes an index of length `2` over
`Base.OneTo(2)`, and `Index(1:3)` makes one over an explicit range. Each call mints a new
name, so two indices built the same way are still distinct, and tensors share a dimension
only when they share the same `Index`.

# Examples

```jldoctest
julia> i = Index(2);

julia> length(i)
2
```
"""
const Index = NamedUnitRange{IndexName}

# TODO: Define for `NamedViewIndex`.
id(i::Index) = id(name(i))
tags(i::Index) = tags(name(i))
plev(i::Index) = plev(name(i))

# TODO: Define for `NamedViewIndex`.
hastag(i::Index, tagname::String) = hastag(name(i), tagname)

# TODO: Define for `NamedViewIndex`.
gettag(i::Index, tagname::String) = gettag(name(i), tagname)
gettag(i::Index, tagname::String, default) = gettag(name(i), tagname, default)
settag(i::Index, tagname::String, tag::String) = setname(i, settag(name(i), tagname, tag))
unsettag(i::Index, tagname::String) = setname(i, unsettag(name(i), tagname))

setplev(i::Index, plev) = setname(i, setplev(name(i), plev))
prime(i::Index) = setname(i, prime(name(i)))
noprime(i::Index) = setname(i, noprime(name(i)))

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
    lenstr = "length=$(length(i))"
    idstr = "|id=$(shortid(id(i)))"
    tagsstr = !isempty(tags(i)) ? "|$(tagsstring(tags(i)))" : ""
    primestr = primestring(plev(i))
    str = "Index($(lenstr)$(idstr)$(tagsstr))$(primestr)"
    print(io, str)
    return nothing
end
