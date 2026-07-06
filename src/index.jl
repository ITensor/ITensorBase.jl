using Accessors: @set
using Random: AbstractRNG, RandomDevice
using TensorAlgebra: TensorAlgebra as TA
using UUIDs: UUID, uuid4

tagpairstring(pair::Pair) = string(first(pair)) * "=>" * string(last(pair))
function tagsstring(tags)
    tagpairs = collect(tags)  # SortedDict iterates in sorted-key order
    tagpair1, tagpair_rest = Iterators.peel(tagpairs)
    return mapreduce(*, tagpair_rest; init = tagpairstring(tagpair1)) do tagpair
        return "," * tagpairstring(tagpair)
    end
end

"""
    IndexName

The name carried by an [`Index`](@ref): a freshly minted unique identifier together with a set
of tags and an integer prime level. Two `IndexName`s compare equal only when their
identifier, tags, and prime level all match, so independently constructed indices stay
distinct. [`prime`](@ref) raises the prime level and [`noprime`](@ref) resets it. `IndexName`
is the dimension-name type behind the legacy ITensor surface, where `Index` is
`NamedUnitRange{IndexName}` and [`ITensor`](@ref) is `NamedTensor{IndexName}`.
"""
struct IndexName <: AbstractName
    id::UUID
    tags::SortedDict{Symbol, Symbol}
    plev::Int
end
function IndexName(
        rng::AbstractRNG = RandomDevice(); id::UUID = uuid4(rng),
        tags = (), plev::Int = 0
    )
    return IndexName(id, to_tags(tags), plev)
end
# Minting from an existing name deliberately drops its tags and prime level (the generic
# name-instance fallback routes through this type form): a fresh name (a factorization's
# new bond, an operator's fresh codomain) has no relationship to the seed's decoration,
# only to its name type.
function uniquename(rng::AbstractRNG, ::Type{<:IndexName})
    return IndexName(rng)
end

# Derive contractions on integer labels: an `IndexName` carries an id and a tag dictionary and is
# far costlier to compare than an integer, and deriving a contraction makes several comparison
# passes over the labels. See `TensorAlgebra.label_type`.
TA.label_type(::Type{<:IndexName}) = Int

to_symbol_pair(p::Pair) = Symbol(first(p)) => Symbol(last(p))

# Like `Dict`, accept one or more bare `Pair`s as tags. A `Pair` iterates over
# its two elements, so it can't fall through to the collection method below.
to_tags(ps::Pair...) = to_tags(ps)
to_tags(tags) = SortedDict{Symbol, Symbol}(to_symbol_pair(p) for p in tags)

id(n::IndexName) = getfield(n, :id)

# Internal: the stored tags as `Symbol => Symbol`, used by the hot comparison,
# hashing, and display paths. `tags` is the public string-valued view of this.
tags_stored(n::IndexName) = getfield(n, :tags)

"""
    tags(i)

Return the tags of an index or index name as an `AbstractDict` mapping tag names to
tag values, both `AbstractString`s.

The concrete dictionary type and string type are implementation details and may
change.
"""
function tags(n::IndexName)
    return SortedDict{String, String}(
        String(k) => String(v) for (k, v) in tags_stored(n)
    )
end

plev(n::IndexName) = getfield(n, :plev)

function Base.:(==)(n1::IndexName, n2::IndexName)
    return id(n1) == id(n2) && plev(n1) == plev(n2) && tags_stored(n1) == tags_stored(n2)
end
function Base.isequal(n1::IndexName, n2::IndexName)
    return isequal(id(n1), id(n2)) &&
        isequal(plev(n1), plev(n2)) &&
        isequal(tags_stored(n1), tags_stored(n2))
end
function Base.isless(n1::IndexName, n2::IndexName)
    t1 = (id(n1), plev(n1), keys(tags_stored(n1)), values(tags_stored(n1)))
    t2 = (id(n2), plev(n2), keys(tags_stored(n2)), values(tags_stored(n2)))
    return isless(t1, t2)
end
function Base.hash(n::IndexName, h::UInt)
    h = hash(:IndexName, h)
    h = hash(id(n), h)
    h = hash(plev(n), h)
    h = hash(tags_stored(n), h)
    return h
end

setid(n::IndexName, id) = @set n.id = id
settags(n::IndexName, tags) = @set n.tags = tags
setplev(n::IndexName, plev) = @set n.plev = plev

hastag(n::IndexName, tagname) = haskey(tags_stored(n), Symbol(tagname))

gettag(n::IndexName, tagname) = String(tags_stored(n)[Symbol(tagname)])
function gettag(n::IndexName, tagname, default)
    t = tags_stored(n)
    k = Symbol(tagname)
    return haskey(t, k) ? String(t[k]) : default
end
function settag(n::IndexName, tagname, tag)
    newtags = copy(tags_stored(n))
    newtags[Symbol(tagname)] = Symbol(tag)
    return settags(n, newtags)
end
function unsettag(n::IndexName, tagname)
    newtags = copy(tags_stored(n))
    delete!(newtags, Symbol(tagname))
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
    tagsstr = !isempty(tags_stored(i)) ? "|$(tagsstring(tags_stored(i)))" : ""
    primestr = primestring(plev(i))
    str = "IndexName($(idstr)$(tagsstr))$(primestr)"
    print(io, str)
    return nothing
end

"""
    Index(space)

An index of an [`ITensor`](@ref): a named unit range whose name is an [`IndexName`](@ref), a
freshly minted, unique identifier carrying tags and a prime level. The argument is a space that is converted to a
range: `Index(2)` makes an index of length `2` over `Base.OneTo(2)`,
`Index(1:3)` makes one over an explicit range, and (with GradedArrays loaded)
`Index([U1(0) => 2, U1(1) => 3])` makes one over a graded range. Each call mints a new
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

# `IndexName`-specialized aliases for the named-dims tensor hierarchy. The
# name-generic primaries are defined earlier (`abstractnamedtensor.jl`,
# `namedtensor.jl`, `namedtensoroperator.jl`); these fix the dimname flavor to
# `IndexName`, recovering the legacy ITensor surface. They live here because they
# reference `IndexName`, just like `Index` itself.

"""
    AbstractITensor

Alias for `AbstractNamedTensor{IndexName}`: the [`AbstractNamedTensor`](@ref)
supertype with dimension names fixed to [`IndexName`](@ref) (the names carried by
[`Index`](@ref)).
"""
const AbstractITensor = AbstractNamedTensor{IndexName}

"""
    ITensor

Alias for `NamedTensor{IndexName}`: a [`NamedTensor`](@ref) whose dimension
names are [`IndexName`](@ref)s, the names carried by [`Index`](@ref). This is the legacy
ITensor type. Use [`NamedTensor`](@ref) for the dimname-flavor-generic type.
"""
const ITensor = NamedTensor{IndexName}

const ITensorOperator = NamedTensorOperator{IndexName}

# TODO: Define for `NamedViewIndex`.
id(i::Index) = id(name(i))
tags_stored(i::Index) = tags_stored(name(i))
tags(i::Index) = tags(name(i))
plev(i::Index) = plev(name(i))

# TODO: Define for `NamedViewIndex`.
hastag(i::Index, tagname) = hastag(name(i), tagname)

# TODO: Define for `NamedViewIndex`.
gettag(i::Index, tagname) = gettag(name(i), tagname)
gettag(i::Index, tagname, default) = gettag(name(i), tagname, default)
settag(i::Index, tagname, tag) = setname(i, settag(name(i), tagname, tag))
unsettag(i::Index, tagname) = setname(i, unsettag(name(i), tagname))

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
    tagsstr = !isempty(tags_stored(i)) ? "|$(tagsstring(tags_stored(i)))" : ""
    primestr = primestring(plev(i))
    str = "Index($(lenstr)$(idstr)$(tagsstr))$(primestr)"
    print(io, str)
    return nothing
end
