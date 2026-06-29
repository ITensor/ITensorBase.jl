# Modeled on TensorKit.jl's `SortedVectorDict`, an `AbstractDict` backed by two
# parallel vectors kept sorted by key. Comparable designs include
# DataStructures.jl's `SortedDict` (backed by a balanced tree) and
# Dictionaries.jl's `ArrayDictionary` (parallel arrays).
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

function Base.haskey(d::SortedDict, key)
    # Linear scan. Use searchsortedfirst on the sorted keys for large collections.
    return !isnothing(findfirst(isequal(key), keys(d)))
end
function Base.getindex(d::SortedDict, key)
    # Linear scan. Use searchsortedfirst on the sorted keys for large collections.
    i = findfirst(isequal(key), keys(d))
    isnothing(i) && throw(KeyError(key))
    return @inbounds values(d)[i]
end
function Base.get(d::SortedDict, key, default)
    # Linear scan. Use searchsortedfirst on the sorted keys for large collections.
    i = findfirst(isequal(key), keys(d))
    return isnothing(i) ? default : @inbounds values(d)[i]
end
function Base.setindex!(d::SortedDict, value, key)
    # Linear scan. Use searchsortedfirst on the sorted keys for large collections.
    i = findfirst(isequal(key), keys(d))
    if isnothing(i)
        # Insertion point so the keys stay sorted. Linear scan, same upgrade as above.
        i = something(findfirst(>(key), keys(d)), length(d) + 1)
        insert!(keys(d), i, key)
        insert!(values(d), i, value)
    else
        @inbounds values(d)[i] = value
    end
    return d
end
function Base.delete!(d::SortedDict, key)
    # Linear scan. Use searchsortedfirst on the sorted keys for large collections.
    i = findfirst(isequal(key), keys(d))
    if !isnothing(i)
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
