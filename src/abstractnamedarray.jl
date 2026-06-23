# `Name` leads (matching `AbstractITensor{DimName}`); `UnnamedT` is the unwrapped
# element type and `N` the rank. The element type is always `Named{Name, UnnamedT}`,
# so it is hardcoded in the `AbstractArray` supertype rather than carried as a
# parameter. The wrapped-container type lives only on the concrete subtypes.
abstract type AbstractNamedArray{Name, UnnamedT, N} <:
AbstractArray{Named{Name, UnnamedT}, N} end

const AbstractNamedVector{Name, UnnamedT} = AbstractNamedArray{Name, UnnamedT, 1}
const AbstractNamedMatrix{Name, UnnamedT} = AbstractNamedArray{Name, UnnamedT, 2}

# Minimal interface.
unnamed(a::AbstractNamedArray) = throw(MethodError(unnamed, Tuple{typeof(a)}))
name(a::AbstractNamedArray) = throw(MethodError(name, Tuple{typeof(a)}))

# This can be customized to output different named array types,
# such as `namedarray(a::AbstractArray, name::IndexName) = Index(a, name)`.
namedarray(a::AbstractArray, name) = NamedArray(a, name)

# Shorthand.
named(a::AbstractArray, name) = namedarray(a, name)

# Derived interface.
# TODO: Use `Accessors.@set`?
setname(a::AbstractNamedArray, name) = namedarray(unnamed(a), name)

# `Name` leads, so `nametype` reads it from the abstract type. The wrapped
# container type lives only on the concrete subtypes, so `unnamedtype` is defined
# per concrete type rather than here.
nametype(::Type{<:AbstractNamedArray{Name}}) where {Name} = Name

# Traits.
isnamed(::Type{<:AbstractNamedArray}) = true

# Equality and hashing are type-agnostic across named array types, following Base's
# array convention (`[1, 2, 3] == 1:3`, and they hash equally): two named arrays are
# equal when their names and unnamed values are equal, regardless of concrete type.
# Hashing uses a single shared tag (not the concrete type) so that
# `a == b => hash(a) == hash(b)` holds; there are no external subtypes that need to
# override this.
function Base.:(==)(a1::AbstractNamedArray, a2::AbstractNamedArray)
    return name(a1) == name(a2) && unnamed(a1) == unnamed(a2)
end
Base.hash(a::AbstractNamedArray, h::UInt) = hash_named(:NamedArray, a, h)

getindex_named(a::AbstractArray, I...) = named(getindex(unnamed(a), I...), name(a))

# Array funcionality.
Base.size(a::AbstractNamedArray) = map(s -> named(s, name(a)), size(unnamed(a)))
Base.axes(a::AbstractNamedArray) = map(s -> named(s, name(a)), axes(unnamed(a)))
Base.eachindex(a::AbstractNamedArray) = eachindex(unnamed(a))
# A named array carries a single name, so its length is that name attached to the
# unnamed length. No fusion is involved, unlike a multi-dim `AbstractITensor`, which
# has no single name and so does not define `length`.
Base.length(a::AbstractNamedArray) = named(length(unnamed(a)), name(a))
function Base.getindex(a::AbstractNamedArray{<:Any, <:Any, N}, I::Vararg{Int, N}) where {N}
    return getindex_named(a, I...)
end
function Base.getindex(a::AbstractNamedArray, I::Int)
    return getindex_named(a, I)
end
Base.isempty(a::AbstractNamedArray) = isempty(unnamed(a))

## function Base.AbstractArray{Int}(a::AbstractNamedArray)
##   return AbstractArray{Int}(unnamed(a))
## end
##
## Base.iterate(a::AbstractNamedArray) = isempty(a) ? nothing : (first(a), first(a))
## function Base.iterate(a::AbstractNamedArray, i)
##   i == last(a) && return nothing
##   next = named(unnamed(i) + unnamed(step(a)), name(a))
##   return (next, next)
## end

function uniquename(rng::AbstractRNG, a::AbstractNamedArray)
    return named(unnamed(a), uniquename(rng, name(a)))
end

function Base.show(io::IO, a::AbstractNamedArray)
    print(io, "named(", unnamed(a), ", ", repr(name(a)), ")")
    return nothing
end
function Base.show(io::IO, mime::MIME"text/plain", a::AbstractNamedArray)
    print(io, "named(\n")
    show(io, mime, unnamed(a))
    print(io, ",\n ", repr(name(a)), ")")
    return nothing
end
