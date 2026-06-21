# `Name` leads (matching `AbstractITensor{DimName}`); `DenamedT` is the unwrapped
# element type and `N` the rank. The element type is always `Named{Name, DenamedT}`,
# so it is hardcoded in the `AbstractArray` supertype rather than carried as a
# parameter. The wrapped-container type lives only on the concrete subtypes.
abstract type AbstractNamedArray{Name, DenamedT, N} <:
AbstractArray{Named{Name, DenamedT}, N} end

const AbstractNamedVector{Name, DenamedT} = AbstractNamedArray{Name, DenamedT, 1}
const AbstractNamedMatrix{Name, DenamedT} = AbstractNamedArray{Name, DenamedT, 2}

# Minimal interface.
denamed(a::AbstractNamedArray) = throw(MethodError(denamed, Tuple{typeof(a)}))
name(a::AbstractNamedArray) = throw(MethodError(name, Tuple{typeof(a)}))

# This can be customized to output different named array types,
# such as `namedarray(a::AbstractArray, name::IndexName) = Index(a, name)`.
namedarray(a::AbstractArray, name) = NamedArray(a, name)

# Shorthand.
named(a::AbstractArray, name) = namedarray(a, name)

# Derived interface.
# TODO: Use `Accessors.@set`?
setname(a::AbstractNamedArray, name) = namedarray(denamed(a), name)

# `Name` leads, so `nametype` reads it from the abstract type. The wrapped
# container type lives only on the concrete subtypes, so `denamedtype` is defined
# per concrete type rather than here.
nametype(::Type{<:AbstractNamedArray{Name}}) where {Name} = Name

# Traits.
isnamed(::Type{<:AbstractNamedArray}) = true

# Equality and hashing are type-agnostic across named array types, following Base's
# array convention (`[1, 2, 3] == 1:3`, and they hash equally): two named arrays are
# equal when their names and denamed values are equal, regardless of concrete type.
# Hashing uses a single shared tag (not the concrete type) so that
# `a == b => hash(a) == hash(b)` holds; there are no external subtypes that need to
# override this.
function Base.:(==)(a1::AbstractNamedArray, a2::AbstractNamedArray)
    return name(a1) == name(a2) && denamed(a1) == denamed(a2)
end
Base.hash(a::AbstractNamedArray, h::UInt) = hash_named(:NamedArray, a, h)

getindex_named(a::AbstractArray, I...) = named(getindex(denamed(a), I...), name(a))

# Array funcionality.
Base.size(a::AbstractNamedArray) = map(s -> named(s, name(a)), size(denamed(a)))
Base.axes(a::AbstractNamedArray) = map(s -> named(s, name(a)), axes(denamed(a)))
Base.eachindex(a::AbstractNamedArray) = eachindex(denamed(a))
function Base.getindex(a::AbstractNamedArray{<:Any, <:Any, N}, I::Vararg{Int, N}) where {N}
    return getindex_named(a, I...)
end
function Base.getindex(a::AbstractNamedArray, I::Int)
    return getindex_named(a, I)
end
Base.isempty(a::AbstractNamedArray) = isempty(denamed(a))

## function Base.AbstractArray{Int}(a::AbstractNamedArray)
##   return AbstractArray{Int}(denamed(a))
## end
##
## Base.iterate(a::AbstractNamedArray) = isempty(a) ? nothing : (first(a), first(a))
## function Base.iterate(a::AbstractNamedArray, i)
##   i == last(a) && return nothing
##   next = named(denamed(i) + denamed(step(a)), name(a))
##   return (next, next)
## end

function uniquename(rng::AbstractRNG, a::AbstractNamedArray)
    return named(denamed(a), uniquename(rng, name(a)))
end

function Base.show(io::IO, a::AbstractNamedArray)
    print(io, "named(", denamed(a), ", ", repr(name(a)), ")")
    return nothing
end
function Base.show(io::IO, mime::MIME"text/plain", a::AbstractNamedArray)
    print(io, "named(\n")
    show(io, mime, denamed(a))
    print(io, ",\n ", repr(name(a)), ")")
    return nothing
end
