using Accessors: @set
using Random: AbstractRNG, RandomDevice
using TensorAlgebra: TensorAlgebra as TA
using UUIDs: UUID, uuid4

# A tag with an empty value is a bare label: print just its key, so `"i" => ""` shows as `i`.
function tagpairstring(pair::Pair)
    key, value = string(first(pair)), string(last(pair))
    return isempty(value) ? key : key * "=>" * value
end
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

# A bare label (a `String` or `Symbol` with no value) is a tag with an empty value. A lone one
# is a single tag, not a collection to iterate over (a `String` would iterate into its `Char`s),
# so wrap it in a one-element tuple like the varargs `Pair` method above.
for T in (AbstractString, Symbol)
    @eval to_symbol_pair(tag::$T) = Symbol(tag) => Symbol("")
    @eval to_tags(tag::$T) = to_tags((tag,))
end

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
are given as one or more `key => value` pairs, bare labels (a `String` or `Symbol`, taken as
a tag with an empty value), a collection mixing these, or an `AbstractDict`; keys and values
may be `String`s or `Symbol`s. See also [`unsettags`](@ref), [`emptytags`](@ref).
"""
settags(n::IndexName, ps::Pair...) = settags(n, ps)
# A lone `Pair` or bare label iterates over its elements, so these single-tag methods need to
# exist rather than letting one fall through to the `for p in ps` loop below (cf. `to_tags`).
for T in (AbstractString, Symbol)
    @eval settags(n::IndexName, tag::$T) = settags(n, (tag,))
end
function settags(n::IndexName, ps)
    for p in ps
        k, v = to_symbol_pair(p)
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
    prime(i, plinc = 1)
    prime(a::AbstractNamedTensor, plinc = 1)
    prime(a::AbstractNamedTensor, is)
    prime(a::AbstractNamedTensor, plinc, is)
    prime(predicate, a::AbstractNamedTensor, plinc = 1)

Increment the prime level of an index or index name by `plinc`, returning a new index that
is distinct from `i`. Priming is the usual way to make a second copy of an index that
carries the same tags but is not contracted against the original. The inverse is
[`noprime`](@ref), which resets the prime level to zero. Given a tensor, prime all of its
indices.

Given a tensor and `is`, prime only those indices, leaving the rest alone. `is` is a
collection, or a single index or index name, and `plinc` goes before it. Given a predicate
instead, prime the indices of `a` for which `predicate` is true, where `plinc` trails the
predicate because the predicate has to come first. An index of `a` is selected
by its full name, prime level included, so `noprime(prime(a, i), i)` leaves `a` unchanged:
the tensor holds `i'`, not `i`. Name the level you mean with `prime(i, 2)`. An index that `a`
does not have is ignored.

A negative `plinc` lowers the prime level, so `prime(a, -1)` undoes one level of priming.
Nothing clamps at zero: going below it yields a negative prime level, which is legal but
prints a warning.

# Examples

```jldoctest
julia> i = Index(2);

julia> prime(i) == i
false

julia> noprime(prime(i)) == i
true

julia> prime(i, 2) == prime(prime(i))
true

julia> j = Index(3);

julia> a = NamedTensor(zeros(2, 3), (i, j));

julia> inds(prime(a, i)) == [prime(i), j]
true

julia> inds(prime(a, 2, i)) == [prime(i, 2), j]
true

julia> prime(prime(a, 2), -2) == a
true

julia> inds(prime(n -> ITensorBase.plev(n) == 0, prime(a, i))) == [prime(i), prime(j)]
true
```

See also [`noprime`](@ref), [`sim`](@ref), [`Index`](@ref).
"""
function prime end

"""
    noprime(i)
    noprime(a::AbstractNamedTensor)
    noprime(a::AbstractNamedTensor, is)
    noprime(predicate, a::AbstractNamedTensor)

Reset the prime level of an index or index name to zero, returning a new index. This
undoes any number of [`prime`](@ref) calls. Given a tensor, reset the prime level of all of
its indices.

Given a tensor and `is`, reset only those indices, leaving the rest alone. `is` is a
collection, or a single index or index name. Given a predicate instead, reset the indices of
`a` for which `predicate` is true, which is the way to select on something other than the
exact name, such as a tag. Selection is by full name, so the index passed must carry the
prime level it has on `a`, and `noprime(a, i)` does nothing to a tensor holding `i'`.

# Examples

```jldoctest
julia> i = Index(2);

julia> noprime(prime(i)) == i
true

julia> j = Index(3);

julia> a = prime(NamedTensor(zeros(2, 3), (i, j)));

julia> inds(noprime(a, prime(j))) == [prime(i), j]
true

julia> s = ITensorBase.settags(Index(2), "Site");

julia> b = prime(NamedTensor(zeros(2, 3), (s, j)));

julia> inds(noprime(x -> ITensorBase.hastag(x, "Site"), b)) == [s, prime(j)]
true
```

See also [`prime`](@ref), [`sim`](@ref), [`Index`](@ref).
"""
function noprime end

"""
    sim(i)
    sim(a::AbstractNamedTensor)
    sim(a::AbstractNamedTensor, is)
    sim(predicate, a::AbstractNamedTensor)

Return a "similar" index: a new index (or, given a tensor, a tensor with all of its indices
replaced) carrying the same tags and prime level as `i` but a fresh unique identifier, so it
is distinct from `i` and will not contract against it. This is the index-manipulation
spelling of [`uniquename`](@ref) on an index.

Given a tensor and `is`, replace only those indices, leaving the rest alone. `is` is a
collection, or a single index or index name. Given a predicate instead, replace the indices
of `a` for which `predicate` is true. Selection is by full name, prime level included, and an
index that `a` does not have is ignored.

# Examples

```jldoctest
julia> i = Index(2);

julia> sim(i) == i
false

julia> length(sim(i))
2

julia> j = Index(3);

julia> a = NamedTensor(zeros(2, 3), (i, j));

julia> inds(sim(a, i))[1] == i
false

julia> inds(sim(a, i))[2] == j
true
```

See also [`uniquename`](@ref), [`prime`](@ref), [`noprime`](@ref).
"""
function sim end

prime(n::IndexName, plinc::Integer = 1) = setplev(n, plev(n) + plinc)
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
and default to no tags and prime level `0`. `tags` accepts the same inputs as [`settags`](@ref):
a `key => value` pair, a bare label like `"i"` (a `String` or `Symbol`, taken as a tag with an
empty value), a collection mixing these, or an `AbstractDict`.

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
prime(i::Index, plinc::Integer = 1) = setname(i, prime(name(i), plinc))
noprime(i::Index) = setname(i, noprime(name(i)))
sim(i::Index) = setname(i, sim(name(i)))

# Whole-tensor index manipulation: relabel every index name-only via `rename`, leaving the
# data and spaces untouched.
#
# The selected forms take a collection of indices, so a lone index needs methods of its own:
# an `Index` is a `NamedUnitRange`, which iterates its own elements, and would otherwise be
# read as a collection of integers rather than as one index.
for f in [:prime, :noprime, :sim]
    @eval begin
        $f(a::AbstractNamedTensor) = rename($f, a)
        $f(a::AbstractNamedTensor, i::AbstractName) = $f(a, (i,))
        $f(a::AbstractNamedTensor, i::AbstractNamedArray) = $f(a, (i,))
        function $f(a::AbstractNamedTensor, is)
            return rename(a, (name(i) => $f(name(i)) for i in is)...)
        end
        # `predicate` is typed, since an untyped first argument would be ambiguous against the
        # collection form above whenever both arguments are tensors.
        function $f(predicate::Function, a::AbstractNamedTensor)
            return $f(a, Iterators.filter(predicate, inds(a)))
        end
    end
end

# Only `prime` counts levels, so only `prime` takes an increment. It precedes the selection,
# as in `prime(i, plinc)` and in both earlier ITensor generations. `Integer` is disjoint from
# the name and index types above, so this adds no ambiguity. A negative `plinc` lowers the
# level, which is what an `unprime` would do.
prime(a::AbstractNamedTensor, plinc::Integer) = rename(n -> prime(n, plinc), a)
prime(a::AbstractNamedTensor, plinc::Integer, i::AbstractName) = prime(a, plinc, (i,))
prime(a::AbstractNamedTensor, plinc::Integer, i::AbstractNamedArray) = prime(a, plinc, (i,))
function prime(a::AbstractNamedTensor, plinc::Integer, is)
    return rename(a, (name(i) => prime(name(i), plinc) for i in is)...)
end
# `plinc` trails the predicate but precedes a selection. The predicate has to come first, so
# there is nowhere else for it to go, and this is the order legacy settled on too.
function prime(predicate::Function, a::AbstractNamedTensor, plinc::Integer)
    return prime(a, plinc, Iterators.filter(predicate, inds(a)))
end

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

# The space, so a graded index shows its sectors and its arrow rather than just a total length.
# A compact context asks for the length instead, which is what a tensor's summary line uses to
# stay readable with one entry per leg.
function Base.show(io::IO, i::Index)
    sp = space(i)
    # A dual index prints as `dual` of the non-dual one, which is the call that makes it, rather
    # than as a `dual` around the space, which is not a call at all once the space is a vector of
    # `sector => multiplicity` pairs.
    nondual = TA.isdual(sp) ? TA.dual(sp) : sp
    spacestr = if get(io, :compact, false)
        "length=$(length(i))"
    else
        # The space the `Index` was written with rather than the range it stores. `:typeinfo`
        # drops the element-type prefix a vector of pairs would otherwise carry.
        spec = from_range(nondual)
        sprint(show, spec; context = IOContext(io, :typeinfo => typeof(spec)))
    end
    idstr = "|id=$(shortid(uuid(i)))"
    tagsstr = !isempty(tags_stored(i)) ? "|$(tagsstring(tags_stored(i)))" : ""
    primestr = primestring(plev(i))
    str = "Index($(spacestr)$(idstr)$(tagsstr))$(primestr)"
    TA.isdual(sp) && (str = "dual($(str))")
    print(io, str)
    return nothing
end
