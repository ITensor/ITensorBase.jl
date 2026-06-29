"""
    SortedDict{K,V} <: AbstractDict{K,V}

An associative container backed by two parallel `Vector`s kept sorted by key.
Lookup is a linear scan, which is fastest for the small key counts this is used
for (index-name tags). Equality and hashing are structural over the sorted
vectors, so they are cheap and order-independent by construction.
"""
struct SortedDict{K, V} <: AbstractDict{K, V}
    keys::Vector{K}
    values::Vector{V}
    function SortedDict{K, V}(keys::Vector{K}, values::Vector{V}) where {K, V}
        @assert length(keys) == length(values)
        @assert issorted(keys)
        return new{K, V}(keys, values)
    end
end

SortedDict{K, V}() where {K, V} = SortedDict{K, V}(K[], V[])
function SortedDict{K, V}(pairs::Vector{Pair{K, V}}) where {K, V}
    sorted = sort(pairs; by = first)
    return SortedDict{K, V}(map(first, sorted), map(last, sorted))
end
function SortedDict{K, V}(itr) where {K, V}
    return SortedDict{K, V}(Pair{K, V}[Pair{K, V}(k, v) for (k, v) in itr])
end
SortedDict{K, V}(pairs::Pair...) where {K, V} = SortedDict{K, V}(collect(Pair{K, V}, pairs))

Base.length(d::SortedDict) = length(getfield(d, :keys))
Base.keys(d::SortedDict) = getfield(d, :keys)
Base.values(d::SortedDict) = getfield(d, :values)

function Base.iterate(d::SortedDict, i = 1)
    i > length(d) && return nothing
    return (keys(d)[i] => values(d)[i]), i + 1
end

function Base.copy(d::SortedDict{K, V}) where {K, V}
    return SortedDict{K, V}(copy(keys(d)), copy(values(d)))
end

# Linear scan: index of the first key not less than `key`. Fastest for small
# `length(d)`; swap in `searchsortedfirst` if a large-`n` use case appears.
function _searchsortedfirst(ks::Vector, key)
    i = 1
    @inbounds while i <= length(ks) && isless(ks[i], key)
        i += 1
    end
    return i
end

function Base.haskey(d::SortedDict{K}, key) where {K}
    k = convert(K, key)
    i = _searchsortedfirst(keys(d), k)
    return i <= length(d) && isequal(keys(d)[i], k)
end
function Base.getindex(d::SortedDict{K}, key) where {K}
    k = convert(K, key)
    i = _searchsortedfirst(keys(d), k)
    (i <= length(d) && isequal(keys(d)[i], k)) || throw(KeyError(key))
    return @inbounds values(d)[i]
end
function Base.get(d::SortedDict{K}, key, default) where {K}
    k = convert(K, key)
    i = _searchsortedfirst(keys(d), k)
    return (i <= length(d) && isequal(keys(d)[i], k)) ? (@inbounds values(d)[i]) : default
end
function Base.setindex!(d::SortedDict{K, V}, value, key) where {K, V}
    k = convert(K, key)
    v = convert(V, value)
    i = _searchsortedfirst(keys(d), k)
    if i <= length(d) && isequal(keys(d)[i], k)
        @inbounds values(d)[i] = v
    else
        insert!(keys(d), i, k)
        insert!(values(d), i, v)
    end
    return d
end
function Base.delete!(d::SortedDict{K}, key) where {K}
    k = convert(K, key)
    i = _searchsortedfirst(keys(d), k)
    if i <= length(d) && isequal(keys(d)[i], k)
        deleteat!(keys(d), i)
        deleteat!(values(d), i)
    end
    return d
end

# Structural equality/hash over the (already sorted) vectors: fast and exact,
# overriding the slower order-independent `AbstractDict` fallbacks.
function Base.:(==)(d1::SortedDict, d2::SortedDict)
    return keys(d1) == keys(d2) && values(d1) == values(d2)
end
function Base.isequal(d1::SortedDict, d2::SortedDict)
    return isequal(keys(d1), keys(d2)) && isequal(values(d1), values(d2))
end
function Base.hash(d::SortedDict, h::UInt)
    return hash(values(d), hash(keys(d), hash(:SortedDict, h)))
end
