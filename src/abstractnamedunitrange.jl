abstract type AbstractNamedUnitRange{Name, DenamedT <: Integer} <:
AbstractNamedVector{Name, DenamedT} end

# Minimal interface.
denamed(r::AbstractNamedUnitRange) = throw(MethodError(denamed, Tuple{typeof(r)}))
name(r::AbstractNamedUnitRange) = throw(MethodError(name, Tuple{typeof(r)}))

# This can be customized to output different named integer types,
# such as `namedunitrange(r::AbstractUnitRange, name::IndexName) = Index(r, name)`.
namedunitrange(r::AbstractUnitRange, name) = NamedUnitRange(r, name)

# Shorthand.
named(r::AbstractUnitRange, name) = namedunitrange(r, name)

# Derived interface. `setname` differs from the `AbstractNamedArray` method: it
# rebuilds through `named` so the result stays a named unit range, not a named
# array. The rest of the named interface (`==`, `hash`, `isnamed`, `denamedtype`,
# `nametype`, `randname`, `show`, `isempty`) is inherited from `AbstractNamedArray`.
# TODO: Use `Accessors.@set`?
setname(r::AbstractNamedUnitRange, name) = named(denamed(r), name)

# Forward `conj` to the underlying range so graded axes flip their sector
# arrows. The `Base.conj(::AbstractArray{<:Real}) = x` fallback would
# otherwise short-circuit before the inner range is touched.
Base.conj(r::AbstractNamedUnitRange) = named(conj(denamed(r)), name(r))

# Unit range functionality.
Base.first(r::AbstractNamedUnitRange) = named(first(denamed(r)), name(r))
Base.last(r::AbstractNamedUnitRange) = named(last(denamed(r)), name(r))
Base.length(r::AbstractNamedUnitRange) = named(length(denamed(r)), name(r))
Base.size(r::AbstractNamedUnitRange) = (named(length(denamed(r)), name(r)),)
Base.axes(r::AbstractNamedUnitRange) = (named(only(axes(denamed(r))), name(r)),)
Base.step(r::AbstractNamedUnitRange) = named(step(denamed(r)), name(r))
Base.getindex(r::AbstractNamedUnitRange, I::Int) = getindex_named(r, I)
# Fix ambiguity error.
function Base.getindex(r::AbstractNamedUnitRange, I::AbstractUnitRange{<:Integer})
    return getindex_named(r, I)
end
# Fix ambiguity error.
function Base.getindex(r::AbstractNamedUnitRange, I::Colon)
    return getindex_named(r, I)
end
function Base.getindex(r::AbstractNamedUnitRange, I)
    return getindex_named(r, I)
end
# Fixes `r[begin]`/`r[end]`, since `firstindex` and `lastindex`
# returned named indices.
function Base.getindex(r::AbstractNamedUnitRange, I::NamedInteger)
    @assert name(r) == name(I)
    return getindex_named(r, denamed(I))
end

# Named ranges are not `AbstractUnitRange`s, so `CartesianIndices` over a tuple of
# them has no Base method; dename to the parent ranges so `CartesianIndices` of a
# named tensor matches the parent's.
function Base.CartesianIndices(
        rs::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}}
    )
    return CartesianIndices(denamed.(rs))
end

# Show compactly; the inherited `AbstractNamedArray` text/plain show is multiline.
Base.show(io::IO, ::MIME"text/plain", r::AbstractNamedUnitRange) = show(io, r)

function Base.AbstractUnitRange{Int}(r::AbstractNamedUnitRange)
    return AbstractUnitRange{Int}(denamed(r))
end

Base.oneto(length::NamedInteger) = named(Base.OneTo(denamed(length)), name(length))
namedoneto(length::Integer, name) = Base.oneto(named(length, name))
Base.iterate(r::AbstractNamedUnitRange) = isempty(r) ? nothing : (first(r), first(r))
function Base.iterate(r::AbstractNamedUnitRange, i)
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

function Base.getindex(r::AbstractNamedUnitRange, I::FirstIndex)
    return first(r)
end
function Base.getindex(r::AbstractNamedUnitRange, I::LastIndex)
    return last(r)
end
