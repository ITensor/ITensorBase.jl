struct NamedUnitRange{Name, DenamedT <: Integer, Denamed <: AbstractUnitRange{DenamedT}} <:
    AbstractNamedVector{Name, DenamedT}
    value::Denamed
    name::Name
end

# Minimal interface.
denamed(i::NamedUnitRange) = i.value
name(i::NamedUnitRange) = i.name
denamedtype(::Type{<:NamedUnitRange{<:Any, <:Any, Denamed}}) where {Denamed} = Denamed

# Construct from a range or length, minting a fresh name of the requested flavor.
# Generic over the name type via `uniquename`, so `Index(3)` (with `Index` a
# `NamedUnitRange{IndexName}` alias) needs no `Index`-specific constructor.
function NamedUnitRange{Name}(r::AbstractUnitRange) where {Name}
    return NamedUnitRange(r, uniquename(Name))
end
function NamedUnitRange{Name}(length::Integer) where {Name}
    return NamedUnitRange{Name}(Base.OneTo(length))
end

# This can be customized to output different named unit range types.
namedunitrange(r::AbstractUnitRange, name) = NamedUnitRange(r, name)

# Shorthand: attach an existing name to a range.
named(r::AbstractUnitRange, name) = namedunitrange(r, name)

# Derived interface. `setname` differs from the `AbstractNamedArray` method: it
# rebuilds through `named` so the result stays a named unit range, not a named
# array. The rest of the named interface (`==`, `hash`, `isnamed`, `denamedtype`,
# `nametype`, `uniquename`, `show`, `isempty`) is inherited from `AbstractNamedArray`.
# TODO: Use `Accessors.@set`?
setname(r::NamedUnitRange, name) = named(denamed(r), name)

# Forward `conj` to the underlying range so graded axes flip their sector
# arrows. The `Base.conj(::AbstractArray{<:Real}) = x` fallback would
# otherwise short-circuit before the inner range is touched.
Base.conj(r::NamedUnitRange) = named(conj(denamed(r)), name(r))

# Unit range functionality.
Base.first(r::NamedUnitRange) = named(first(denamed(r)), name(r))
Base.last(r::NamedUnitRange) = named(last(denamed(r)), name(r))
Base.length(r::NamedUnitRange) = named(length(denamed(r)), name(r))
Base.size(r::NamedUnitRange) = (named(length(denamed(r)), name(r)),)
Base.axes(r::NamedUnitRange) = (named(only(axes(denamed(r))), name(r)),)
Base.step(r::NamedUnitRange) = named(step(denamed(r)), name(r))
Base.getindex(r::NamedUnitRange, I::Int) = getindex_named(r, I)
# Fix ambiguity error.
function Base.getindex(r::NamedUnitRange, I::AbstractUnitRange{<:Integer})
    return getindex_named(r, I)
end
# Fix ambiguity error.
function Base.getindex(r::NamedUnitRange, I::Colon)
    return getindex_named(r, I)
end
function Base.getindex(r::NamedUnitRange, I)
    return getindex_named(r, I)
end
# Fixes `r[begin]`/`r[end]`, since `firstindex` and `lastindex`
# returned named indices.
function Base.getindex(r::NamedUnitRange, I::NamedInteger)
    @assert name(r) == name(I)
    return getindex_named(r, denamed(I))
end

# Named ranges are not `AbstractUnitRange`s, so `CartesianIndices` over a tuple of
# them has no Base method; dename to the parent ranges so `CartesianIndices` of a
# named tensor matches the parent's.
function Base.CartesianIndices(
        rs::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return CartesianIndices(denamed.(rs))
end

# Show compactly; the inherited `AbstractNamedArray` text/plain show is multiline.
Base.show(io::IO, ::MIME"text/plain", r::NamedUnitRange) = show(io, r)

function Base.AbstractUnitRange{Int}(r::NamedUnitRange)
    return AbstractUnitRange{Int}(denamed(r))
end

Base.oneto(length::NamedInteger) = named(Base.OneTo(denamed(length)), name(length))
namedoneto(length::Integer, name) = Base.oneto(named(length, name))
Base.iterate(r::NamedUnitRange) = isempty(r) ? nothing : (first(r), first(r))
function Base.iterate(r::NamedUnitRange, i)
    i == last(r) && return nothing
    next = named(denamed(i) + denamed(step(r)), name(r))
    return (next, next)
end

struct NamedColon{Name} <: Function
    name::Name
end
denamed(c::NamedColon) = Colon()
name(c::NamedColon) = c.name
named(::Colon, name) = NamedColon(name)

struct FirstIndex{Arr, Dim}
    array::Arr
    dim::Dim
end
Base.to_index(i::FirstIndex) = denamed(first(axes(i.array, i.dim)))

struct LastIndex{Arr, Dim}
    array::Arr
    dim::Dim
end
Base.to_index(i::LastIndex) = denamed(last(axes(i.array, i.dim)))

function Base.getindex(r::NamedUnitRange, I::FirstIndex)
    return first(r)
end
function Base.getindex(r::NamedUnitRange, I::LastIndex)
    return last(r)
end
