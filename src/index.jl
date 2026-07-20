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
    uuid::UUID
    tags::SortedDict{Symbol, Symbol}
    plev::Int
end
function IndexName(
        rng::AbstractRNG = RandomDevice(); uuid::UUID = uuid4(rng),
        tags = (), plev::Int = 0
    )
    return IndexName(uuid, to_tags(tags), plev)
end
# `uniquename` on an existing `IndexName` keeps its tags and prime level, minting only a
# fresh id (the legacy `sim`). The type form drops them: a factorization bond or a fresh
# operator leg has no relationship to any seed's decoration, so its callers pass the name
# type to opt out of inheriting it.
function uniquename(rng::AbstractRNG, name::IndexName)
    return IndexName(rng; tags = tags_stored(name), plev = plev(name))
end
function uniquename(rng::AbstractRNG, ::Type{<:IndexName}; kwargs...)
    return IndexName(rng; kwargs...)
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

uuid(n::IndexName) = getfield(n, :uuid)

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

"""
    plev(i)

Return the prime level of an index or index name: a non-negative integer raised by
[`prime`](@ref) and reset by [`noprime`](@ref).
"""
plev(n::IndexName) = getfield(n, :plev)

# The tags dictionary is the only costly field to compare, so short-circuit it with `===`:
# a name reused across tensors carries the same tags object and skips the dictionary walk.
function Base.:(==)(n1::IndexName, n2::IndexName)
    return uuid(n1) == uuid(n2) && plev(n1) == plev(n2) &&
        (tags_stored(n1) === tags_stored(n2) || tags_stored(n1) == tags_stored(n2))
end
function Base.isequal(n1::IndexName, n2::IndexName)
    return isequal(uuid(n1), uuid(n2)) && isequal(plev(n1), plev(n2)) &&
        (
        tags_stored(n1) === tags_stored(n2) ||
            isequal(tags_stored(n1), tags_stored(n2))
    )
end
function Base.isless(n1::IndexName, n2::IndexName)
    t1 = (uuid(n1), plev(n1), keys(tags_stored(n1)), values(tags_stored(n1)))
    t2 = (uuid(n2), plev(n2), keys(tags_stored(n2)), values(tags_stored(n2)))
    return isless(t1, t2)
end
function Base.hash(n::IndexName, h::UInt)
    h = hash(:IndexName, h)
    h = hash(uuid(n), h)
    h = hash(plev(n), h)
    h = hash(tags_stored(n), h)
    return h
end

setuuid(n::IndexName, uuid) = @set n.uuid = uuid
setplev(n::IndexName, plev) = @set n.plev = plev

# Internal whole-dictionary install. `settags` is the public merge-semantics verb;
# this is the raw replace behind the single-key `settag`/`unsettag` primitives.
setstoredtags(n::IndexName, tags) = @set n.tags = tags

"""
    hastag(i, key)

Return `true` if the index or index name carries a tag under `key`.
"""
hastag(n::IndexName, tagname) = haskey(tags_stored(n), Symbol(tagname))

"""
    gettag(i, key)
    gettag(i, key, default)

Return the tag value stored under `key` as a `String`. The two-argument form throws if
`key` is absent; the three-argument form returns `default` instead. See also
[`gettags`](@ref).
"""
gettag(n::IndexName, tagname) = String(tags_stored(n)[Symbol(tagname)])
function gettag(n::IndexName, tagname, default)
    t = tags_stored(n)
    k = Symbol(tagname)
    return haskey(t, k) ? String(t[k]) : default
end

"""
    gettags(i, keys)

Return the sub-dictionary of the index's tags whose keys are in `keys`, skipping any that
are absent (so the result never has more keys than requested and never throws). The
dictionary and string types are implementation details. See also [`gettag`](@ref), [`tags`](@ref).
"""
function gettags(n::IndexName, tagnames)
    t = tags_stored(n)
    ks = (Symbol(k) for k in tagnames)
    return SortedDict{String, String}(
        String(k) => String(t[k]) for k in ks if haskey(t, k)
    )
end

# `settag`/`unsettag` are internal single-key primitives; the public plural verbs
# `settags`/`unsettags` are built on them.
function settag(n::IndexName, tagname, tag)
    newtags = copy(tags_stored(n))
    newtags[Symbol(tagname)] = Symbol(tag)
    return setstoredtags(n, newtags)
end
function unsettag(n::IndexName, tagname)
    newtags = copy(tags_stored(n))
    delete!(newtags, Symbol(tagname))
    return setstoredtags(n, newtags)
end

"""
    settags(i, key => value, ...)
    settags(i, pairs)

Return a new index or index name with the given tags inserted or overwritten. This is a
merge: tags under other keys are kept, and a key that already exists is overwritten. Tags
are given as one or more `key => value` pairs, a collection of pairs, or an `AbstractDict`;
keys and values may be `String`s or `Symbol`s. See also [`unsettags`](@ref),
[`emptytags`](@ref).
"""
settags(n::IndexName, ps::Pair...) = settags(n, ps)
# A lone `Pair` iterates over its two elements, so the varargs method above needs to exist
# rather than letting a single pair fall through to the `for (k, v) in ps` loop (cf. `to_tags`).
function settags(n::IndexName, ps)
    for (k, v) in ps
        n = settag(n, k, v)
    end
    return n
end

"""
    unsettags(i, keys)

Return a new index or index name with the tags under each of `keys` removed. Keys that are
not present are ignored, so this never throws. See also [`settags`](@ref), [`emptytags`](@ref).
"""
function unsettags(n::IndexName, tagnames)
    for k in tagnames
        n = unsettag(n, k)
    end
    return n
end

"""
    emptytags(i)

Return a new index or index name with all tags removed.
"""
emptytags(n::IndexName) = setstoredtags(n, empty(tags_stored(n)))

"""
    decoration(i)

Return the decoration of an index or index name as a `NamedTuple` `(; tags, plev)`. Splatting
it into [`uniquename`](@ref) or the [`Index`](@ref) keyword constructor reproduces that
decoration on a freshly minted, unique name, as in `uniquename(IndexName; decoration(i)...)`.
A name that carries no decoration returns an empty `NamedTuple`.
"""
decoration(n) = (;)
decoration(n::IndexName) = (; tags = tags(n), plev = plev(n))

"""
    prime(i)
    prime(t::AbstractNamedTensor)

Increment the prime level of an index or index name by one, returning a new index that
is distinct from `i`. Priming is the usual way to make a second copy of an index that
carries the same tags but is not contracted against the original. The inverse is
[`noprime`](@ref), which resets the prime level to zero. Given a tensor, prime all of its
indices.

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
    noprime(t::AbstractNamedTensor)

Reset the prime level of an index or index name to zero, returning a new index. This
undoes any number of [`prime`](@ref) calls. Given a tensor, reset the prime level of all of
its indices.

# Examples

```jldoctest
julia> i = Index(2);

julia> noprime(prime(i)) == i
true
```

See also [`prime`](@ref), [`Index`](@ref).
"""
function noprime end

"""
    sim(i)
    sim(t::AbstractNamedTensor)

Return a "similar" index: a new index (or, given a tensor, a tensor with all of its indices
replaced) carrying the same tags and prime level as `i` but a fresh unique identifier, so it
is distinct from `i` and will not contract against it. This is the index-manipulation
spelling of [`uniquename`](@ref) on an index.

# Examples

```jldoctest
julia> i = Index(2);

julia> sim(i) == i
false

julia> length(sim(i))
2
```

See also [`uniquename`](@ref), [`prime`](@ref).
"""
function sim end

prime(n::IndexName) = setplev(n, plev(n) + 1)
noprime(n::IndexName) = setplev(n, 0)
sim(n::IndexName) = uniquename(n)

# Show a short prefix of the `UUID` id rather than the full 36-character string,
# enough to disambiguate indices at a glance without dominating the output. A
# leading prefix (here the first hyphen-delimited group) is the usual short-id
# convention, as in git short hashes and Docker short ids.
shortid(uuid::UUID) = first(string(uuid), 8)

function Base.show(io::IO, i::IndexName)
    idstr = "id=$(shortid(uuid(i)))"
    tagsstr = !isempty(tags_stored(i)) ? "|$(tagsstring(tags_stored(i)))" : ""
    primestr = primestring(plev(i))
    str = "IndexName($(idstr)$(tagsstr))$(primestr)"
    print(io, str)
    return nothing
end

"""
    Index(space; tags, plev)

An index of an [`ITensor`](@ref): a named unit range whose name is an [`IndexName`](@ref), a
freshly minted, unique identifier carrying tags and a prime level. The argument is a space that is converted to a
range: `Index(2)` makes an index of length `2` over `Base.OneTo(2)`,
`Index(1:3)` makes one over an explicit range, and (with GradedArrays loaded)
`Index([U1(0) => 2, U1(1) => 3])` makes one over a graded range. Each call mints a new
name, so two indices built the same way are still distinct, and tensors share a dimension
only when they share the same `Index`.

`tags` and `plev` decorate the freshly minted name, as in `Index(2; tags = "i" => "1", plev = 1)`,
and default to no tags and prime level `0`. `tags` accepts the same inputs as [`settags`](@ref)
(a pair, a collection of pairs, or an `AbstractDict`).

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
uuid(i::Index) = uuid(name(i))
tags_stored(i::Index) = tags_stored(name(i))
tags(i::Index) = tags(name(i))
plev(i::Index) = plev(name(i))

# TODO: Define for `NamedViewIndex`.
hastag(i::Index, tagname) = hastag(name(i), tagname)

# TODO: Define for `NamedViewIndex`.
gettag(i::Index, tagname) = gettag(name(i), tagname)
gettag(i::Index, tagname, default) = gettag(name(i), tagname, default)
gettags(i::Index, tagnames) = gettags(name(i), tagnames)
settag(i::Index, tagname, tag) = setname(i, settag(name(i), tagname, tag))
unsettag(i::Index, tagname) = setname(i, unsettag(name(i), tagname))
settags(i::Index, ps::Pair...) = setname(i, settags(name(i), ps...))
settags(i::Index, ps) = setname(i, settags(name(i), ps))
unsettags(i::Index, tagnames) = setname(i, unsettags(name(i), tagnames))
emptytags(i::Index) = setname(i, emptytags(name(i)))
decoration(i::Index) = decoration(name(i))

setplev(i::Index, plev) = setname(i, setplev(name(i), plev))
prime(i::Index) = setname(i, prime(name(i)))
noprime(i::Index) = setname(i, noprime(name(i)))
sim(i::Index) = setname(i, sim(name(i)))

# Whole-tensor index manipulation: relabel every index name-only via `mapinds`, leaving the
# data and spaces untouched.
prime(a::AbstractNamedTensor) = mapinds(prime, a)
noprime(a::AbstractNamedTensor) = mapinds(noprime, a)
sim(a::AbstractNamedTensor) = mapinds(sim, a)

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
    idstr = "|id=$(shortid(uuid(i)))"
    tagsstr = !isempty(tags_stored(i)) ? "|$(tagsstring(tags_stored(i)))" : ""
    primestr = primestring(plev(i))
    str = "Index($(lenstr)$(idstr)$(tagsstr))$(primestr)"
    print(io, str)
    return nothing
end
